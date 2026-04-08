// =============================================================================
// dem_solver.cpp
// 3D Discrete Element Method (DEM) Solver for Spherical Particles
// Translational motion only (no rotational dynamics)
// Features: Serial + OpenMP parallelisation, verification tests,
//           neighbour-search (cell-linked list), profiling timers
//
// Compile (serial):   g++ -O2 -std=c++17 -o dem dem_solver.cpp
// Compile (OpenMP):   g++ -O2 -std=c++17 -fopenmp -o dem dem_solver.cpp
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {}
#endif

// =============================================================================
// SECTION 1: Data Structures
// =============================================================================

// Represents a single spherical particle
struct Particle {
    double x, y, z;       // position
    double vx, vy, vz;    // velocity
    double fx, fy, fz;    // total force
    double mass;
    double radius;
};

// Simulation configuration parameters
struct SimConfig {
    // Domain
    double Lx, Ly, Lz;    // box dimensions

    // Material parameters
    double kn;             // normal spring stiffness
    double gamma_n;        // damping coefficient

    // Gravity
    double gx, gy, gz;    // gravitational acceleration vector

    // Time integration
    double dt;             // timestep
    double t_end;          // final simulation time

    // Output
    int    out_freq;       // write output every N steps
    std::string out_file;  // output filename prefix

    // OpenMP
    int    n_threads;      // number of OpenMP threads

    // Neighbour search
    bool   use_neighbour_search; // toggle cell-linked list
};

// =============================================================================
// SECTION 2: Initialisation
// =============================================================================

// Single particle at rest (used for verification tests)
void init_single_particle(std::vector<Particle>& particles,
                          double x0, double y0, double z0,
                          double vx0 = 0.0, double vy0 = 0.0, double vz0 = 0.0,
                          double mass = 1.0, double radius = 0.05)
{
    particles.clear();
    Particle p;
    p.x = x0; p.y = y0; p.z = z0;
    p.vx = vx0; p.vy = vy0; p.vz = vz0;
    p.fx = p.fy = p.fz = 0.0;
    p.mass = mass;
    p.radius = radius;
    particles.push_back(p);
}

// Random cloud of N particles packed in the domain
// Uses a simple rejection-free lattice + random jitter
void init_random_particles(std::vector<Particle>& particles,
                           int N,
                           const SimConfig& cfg,
                           double radius = 0.05,
                           double mass   = 1.0,
                           unsigned seed = 42)
{
    particles.clear();
    particles.reserve(N);

    // Seed a simple LCG for reproducibility
    unsigned long rng = seed;
    auto rand01 = [&]() -> double {
        rng = rng * 1664525UL + 1013904223UL;
        return (rng & 0x7FFFFFFF) / double(0x7FFFFFFF);
    };

    int placed = 0;
    int max_attempts = N * 200;
    int attempts = 0;

    while (placed < N && attempts < max_attempts) {
        ++attempts;
        double px = radius + rand01() * (cfg.Lx - 2.0 * radius);
        double py = radius + rand01() * (cfg.Ly - 2.0 * radius);
        double pz = radius + rand01() * (cfg.Lz - 2.0 * radius);

        // Overlap check against already-placed particles
        bool overlap = false;
        for (auto& q : particles) {
            double dx = px - q.x, dy = py - q.y, dz = pz - q.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < (radius + q.radius) * 1.01) { overlap = true; break; }
        }
        if (!overlap) {
            Particle p;
            p.x = px; p.y = py; p.z = pz;
            p.vx = p.vy = p.vz = 0.0;
            p.fx = p.fy = p.fz = 0.0;
            p.mass = mass;
            p.radius = radius;
            particles.push_back(p);
            ++placed;
        }
    }

    if (placed < N)
        std::cerr << "[Warning] Only placed " << placed << " / " << N
                  << " particles (domain too crowded).\n";
}

// =============================================================================
// SECTION 3: Force Routines
// =============================================================================

// --- 3a. Zero all forces ---
void zero_forces(std::vector<Particle>& p)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)p.size(); ++i)
        p[i].fx = p[i].fy = p[i].fz = 0.0;
}

// --- 3b. Body force: gravity ---
void add_gravity(std::vector<Particle>& p, const SimConfig& cfg)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)p.size(); ++i) {
        p[i].fx += p[i].mass * cfg.gx;
        p[i].fy += p[i].mass * cfg.gy;
        p[i].fz += p[i].mass * cfg.gz;
    }
}

// --- 3c. Particle–particle contact force (brute-force O(N^2)) ---
// Uses thread-private accumulators then reduces to avoid race conditions
void compute_particle_contacts(std::vector<Particle>& p,
                                const SimConfig& cfg)
{
    int N = (int)p.size();

    // Thread-private force accumulators (delta_f arrays)
    int nthreads;
    #ifdef _OPENMP
        nthreads = omp_get_max_threads();
    #else
        nthreads = 1;
    #endif

    // Allocate per-thread delta arrays: [thread][particle][3]
    std::vector<std::vector<double>> dfx(nthreads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> dfy(nthreads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> dfz(nthreads, std::vector<double>(N, 0.0));

    #pragma omp parallel
    {
        #ifdef _OPENMP
            int tid = omp_get_thread_num();
        #else
            int tid = 0;
        #endif

        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {

                // Relative position r_ij = x_j - x_i
                double rx = p[j].x - p[i].x;
                double ry = p[j].y - p[i].y;
                double rz = p[j].z - p[i].z;

                double dij = std::sqrt(rx*rx + ry*ry + rz*rz);
                if (dij < 1e-15) continue;  // avoid division by zero

                double delta = p[i].radius + p[j].radius - dij;
                if (delta <= 0.0) continue; // no contact

                // Unit normal n_ij
                double nx = rx / dij;
                double ny = ry / dij;
                double nz = rz / dij;

                // Normal relative velocity v_n,ij = (v_j - v_i) . n_ij
                double dvx = p[j].vx - p[i].vx;
                double dvy = p[j].vy - p[i].vy;
                double dvz = p[j].vz - p[i].vz;
                double vn  = dvx*nx + dvy*ny + dvz*nz;

                // Spring-dashpot normal force (no attractive forces)
                double Fn = std::max(0.0, cfg.kn * delta - cfg.gamma_n * vn);

                double fcx = Fn * nx;
                double fcy = Fn * ny;
                double fcz = Fn * nz;

                // Accumulate into thread-private arrays
                dfx[tid][i] += fcx;  dfy[tid][i] += fcy;  dfz[tid][i] += fcz;
                dfx[tid][j] -= fcx;  dfy[tid][j] -= fcy;  dfz[tid][j] -= fcz;
            }
        }
    }

    // Reduce thread contributions into particle forces
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int t = 0; t < nthreads; ++t) {
            p[i].fx += dfx[t][i];
            p[i].fy += dfy[t][i];
            p[i].fz += dfz[t][i];
        }
    }
}

// --- 3d. Particle–particle contact force (cell-linked list, O(N)) ---
// Divides domain into cells of size >= 2*r_max
void compute_particle_contacts_neighbour(std::vector<Particle>& p,
                                          const SimConfig& cfg)
{
    int N = (int)p.size();
    if (N == 0) return;

    // Find maximum radius
    double r_max = 0.0;
    for (auto& pi : p) r_max = std::max(r_max, pi.radius);
    double cell_size = 2.0 * r_max * 1.01; // slightly larger than 2r_max

    int ncx = std::max(1, (int)(cfg.Lx / cell_size));
    int ncy = std::max(1, (int)(cfg.Ly / cell_size));
    int ncz = std::max(1, (int)(cfg.Lz / cell_size));
    double cx = cfg.Lx / ncx;
    double cy = cfg.Ly / ncy;
    double cz = cfg.Lz / ncz;
    int ncells = ncx * ncy * ncz;

    // Assign particles to cells
    std::vector<int> cell_id(N);
    std::vector<int> count(ncells, 0);
    for (int i = 0; i < N; ++i) {
        int ix = std::min((int)(p[i].x / cx), ncx - 1);
        int iy = std::min((int)(p[i].y / cy), ncy - 1);
        int iz = std::min((int)(p[i].z / cz), ncz - 1);
        cell_id[i] = ix + ncx * (iy + ncy * iz);
        count[cell_id[i]]++;
    }

    // Build cell lists (sorted by cell)
    std::vector<int> offset(ncells + 1, 0);
    for (int c = 0; c < ncells; ++c) offset[c + 1] = offset[c] + count[c];
    std::vector<int> sorted(N);
    std::vector<int> tmp(offset.begin(), offset.end());
    for (int i = 0; i < N; ++i) sorted[tmp[cell_id[i]]++] = i;

    int nthreads;
    #ifdef _OPENMP
        nthreads = omp_get_max_threads();
    #else
        nthreads = 1;
    #endif

    std::vector<std::vector<double>> dfx(nthreads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> dfy(nthreads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> dfz(nthreads, std::vector<double>(N, 0.0));

    #pragma omp parallel
    {
        #ifdef _OPENMP
            int tid = omp_get_thread_num();
        #else
            int tid = 0;
        #endif

        #pragma omp for schedule(dynamic, 4) collapse(3)
        for (int ix = 0; ix < ncx; ++ix)
        for (int iy = 0; iy < ncy; ++iy)
        for (int iz = 0; iz < ncz; ++iz)
        {
            int c = ix + ncx * (iy + ncy * iz);
            // Iterate over this cell and 13 unique neighbour cells
            for (int dix = -1; dix <= 1; ++dix)
            for (int diy = -1; diy <= 1; ++diy)
            for (int diz = -1; diz <= 1; ++diz)
            {
                // Only half-shell to avoid double counting
                if (dix*27 + diy*9 + diz*3 < 0) continue;
                if (dix == 0 && diy == 0 && diz == 0) {
                    // Same cell: i < j
                    int jx = ix + dix, jy = iy + diy, jz = iz + diz;
                    int nc = jx + ncx * (jy + ncy * jz);
                    for (int ai = offset[c]; ai < offset[c+1]; ++ai)
                    for (int bi = offset[nc]; bi < offset[nc+1]; ++bi) {
                        int i = sorted[ai], j = sorted[bi];
                        if (i >= j) continue;
                        double rx = p[j].x-p[i].x, ry = p[j].y-p[i].y, rz = p[j].z-p[i].z;
                        double dij = std::sqrt(rx*rx+ry*ry+rz*rz);
                        if (dij < 1e-15) continue;
                        double delta = p[i].radius + p[j].radius - dij;
                        if (delta <= 0.0) continue;
                        double nx=rx/dij, ny=ry/dij, nz=rz/dij;
                        double vn=(p[j].vx-p[i].vx)*nx+(p[j].vy-p[i].vy)*ny+(p[j].vz-p[i].vz)*nz;
                        double Fn = std::max(0.0, cfg.kn*delta - cfg.gamma_n*vn);
                        dfx[tid][i]+=Fn*nx; dfy[tid][i]+=Fn*ny; dfz[tid][i]+=Fn*nz;
                        dfx[tid][j]-=Fn*nx; dfy[tid][j]-=Fn*ny; dfz[tid][j]-=Fn*nz;
                    }
                } else {
                    int jx = ix+dix, jy = iy+diy, jz = iz+diz;
                    if (jx<0||jx>=ncx||jy<0||jy>=ncy||jz<0||jz>=ncz) continue;
                    int nc = jx + ncx * (jy + ncy * jz);
                    for (int ai = offset[c]; ai < offset[c+1]; ++ai)
                    for (int bi = offset[nc]; bi < offset[nc+1]; ++bi) {
                        int i = sorted[ai], j = sorted[bi];
                        double rx=p[j].x-p[i].x, ry=p[j].y-p[i].y, rz=p[j].z-p[i].z;
                        double dij=std::sqrt(rx*rx+ry*ry+rz*rz);
                        if (dij<1e-15) continue;
                        double delta=p[i].radius+p[j].radius-dij;
                        if (delta<=0.0) continue;
                        double nx=rx/dij, ny=ry/dij, nz=rz/dij;
                        double vn=(p[j].vx-p[i].vx)*nx+(p[j].vy-p[i].vy)*ny+(p[j].vz-p[i].vz)*nz;
                        double Fn=std::max(0.0,cfg.kn*delta-cfg.gamma_n*vn);
                        dfx[tid][i]+=Fn*nx; dfy[tid][i]+=Fn*ny; dfz[tid][i]+=Fn*nz;
                        dfx[tid][j]-=Fn*nx; dfy[tid][j]-=Fn*ny; dfz[tid][j]-=Fn*nz;
                    }
                }
            }
        }
    }

    // Reduce
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < nthreads; ++t) {
            p[i].fx += dfx[t][i];
            p[i].fy += dfy[t][i];
            p[i].fz += dfz[t][i];
        }
}

// --- 3e. Wall contact forces ---
// Applies spring-dashpot force for each of the 6 box walls
void compute_wall_contacts(std::vector<Particle>& p, const SimConfig& cfg)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)p.size(); ++i) {
        double ri = p[i].radius;

        // Helper lambda: apply wall force along one axis
        auto wall_force = [&](double pos, double vel, double wall_pos,
                               double normal_dir) -> double {
            // normal_dir = +1 if wall pushes in +axis, -1 otherwise
            double delta = ri - normal_dir * (pos - wall_pos);
            if (delta <= 0.0) return 0.0;
            double vn = normal_dir * vel;  // velocity into wall
            double Fn = std::max(0.0, cfg.kn * delta - cfg.gamma_n * (-vn));
            return normal_dir * Fn;
        };

        // x-walls
        p[i].fx += wall_force(p[i].x, p[i].vx, 0.0,    +1.0); // x=0
        p[i].fx += wall_force(p[i].x, p[i].vx, cfg.Lx, -1.0); // x=Lx

        // y-walls
        p[i].fy += wall_force(p[i].y, p[i].vy, 0.0,    +1.0);
        p[i].fy += wall_force(p[i].y, p[i].vy, cfg.Ly, -1.0);

        // z-walls
        p[i].fz += wall_force(p[i].z, p[i].vz, 0.0,    +1.0);
        p[i].fz += wall_force(p[i].z, p[i].vz, cfg.Lz, -1.0);
    }
}

// =============================================================================
// SECTION 4: Time Integration (semi-implicit Euler)
// =============================================================================

void integrate_particles(std::vector<Particle>& p, const SimConfig& cfg)
{
    double dt = cfg.dt;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)p.size(); ++i) {
        // Velocity update: v^{n+1} = v^n + (F^n / m) * dt
        p[i].vx += (p[i].fx / p[i].mass) * dt;
        p[i].vy += (p[i].fy / p[i].mass) * dt;
        p[i].vz += (p[i].fz / p[i].mass) * dt;

        // Position update: x^{n+1} = x^n + v^{n+1} * dt
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// =============================================================================
// SECTION 5: Diagnostics
// =============================================================================

double compute_kinetic_energy(const std::vector<Particle>& p)
{
    double KE = 0.0;
    #pragma omp parallel for reduction(+:KE) schedule(static)
    for (int i = 0; i < (int)p.size(); ++i) {
        double v2 = p[i].vx*p[i].vx + p[i].vy*p[i].vy + p[i].vz*p[i].vz;
        KE += 0.5 * p[i].mass * v2;
    }
    return KE;
}

int count_contacts(const std::vector<Particle>& p)
{
    int cnt = 0;
    int N = (int)p.size();
    #pragma omp parallel for reduction(+:cnt) schedule(dynamic,32)
    for (int i = 0; i < N-1; ++i)
        for (int j = i+1; j < N; ++j) {
            double dx=p[j].x-p[i].x, dy=p[j].y-p[i].y, dz=p[j].z-p[i].z;
            if (std::sqrt(dx*dx+dy*dy+dz*dz) < p[i].radius+p[j].radius) ++cnt;
        }
    return cnt;
}

// =============================================================================
// SECTION 6: Output
// =============================================================================

// Writes particle positions/velocities in a simple CSV format
// Each snapshot goes to a separate file: <prefix>_<step>.csv
void write_output(const std::vector<Particle>& p, int step,
                  const std::string& prefix)
{
    std::ostringstream fname;
    fname << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".csv";
    std::ofstream f(fname.str());
    f << "id,x,y,z,vx,vy,vz,r,m\n";
    for (int i = 0; i < (int)p.size(); ++i)
        f << i   << ","
          << p[i].x << "," << p[i].y << "," << p[i].z << ","
          << p[i].vx<< ","<< p[i].vy<< ","<< p[i].vz<< ","
          << p[i].radius << "," << p[i].mass << "\n";
}

// Appends a scalar diagnostic record to a running log file
void write_diagnostics(std::ofstream& log, double t, double KE,
                       int contacts, int N)
{
    log << std::fixed << std::setprecision(8)
        << t << "," << KE << "," << contacts << "," << N << "\n";
}

// =============================================================================
// SECTION 7: Main Simulation Loop
// =============================================================================

// Returns total wall-clock time in seconds
double run_simulation(std::vector<Particle>& particles,
                      const SimConfig& cfg,
                      bool log_diag = true)
{
    omp_set_num_threads(cfg.n_threads);

    int total_steps = (int)(cfg.t_end / cfg.dt);

    std::ofstream diag_log;
    if (log_diag) {
        diag_log.open(cfg.out_file + "_diag.csv");
        diag_log << "time,KE,contacts,N\n";
    }

    // ---- Profiling timers ----
    double t_gravity = 0, t_pp = 0, t_wall = 0, t_integ = 0, t_diag = 0;
    auto clk = [](){ return std::chrono::high_resolution_clock::now(); };
    using dur = std::chrono::duration<double>;

    auto t_start_total = clk();

    for (int step = 0; step <= total_steps; ++step) {
        double t_now = step * cfg.dt;

        // --- Zero forces ---
        zero_forces(particles);

        // --- Gravity ---
        auto t0 = clk();
        add_gravity(particles, cfg);
        t_gravity += dur(clk() - t0).count();

        // --- Particle–Particle contacts ---
        t0 = clk();
        if (cfg.use_neighbour_search)
            compute_particle_contacts_neighbour(particles, cfg);
        else
            compute_particle_contacts(particles, cfg);
        t_pp += dur(clk() - t0).count();

        // --- Wall contacts ---
        t0 = clk();
        compute_wall_contacts(particles, cfg);
        t_wall += dur(clk() - t0).count();

        // --- Integration ---
        t0 = clk();
        integrate_particles(particles, cfg);
        t_integ += dur(clk() - t0).count();

        // --- Diagnostics & Output ---
        t0 = clk();
        if (log_diag && step % cfg.out_freq == 0) {
            double KE = compute_kinetic_energy(particles);
            int nc    = (particles.size() <= 500) ? count_contacts(particles) : -1;
            if (diag_log.is_open())
                write_diagnostics(diag_log, t_now, KE, nc, (int)particles.size());
            write_output(particles, step, cfg.out_file);
        }
        t_diag += dur(clk() - t0).count();
    }

    double total_time = dur(clk() - t_start_total).count();

    // ---- Print profiling report ----
    std::cout << "\n===== Profiling Report =====\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Gravity:            " << t_gravity << " s ("
              << 100*t_gravity/total_time << "%)\n";
    std::cout << "  Particle contacts:  " << t_pp      << " s ("
              << 100*t_pp/total_time      << "%)\n";
    std::cout << "  Wall contacts:      " << t_wall    << " s ("
              << 100*t_wall/total_time    << "%)\n";
    std::cout << "  Integration:        " << t_integ   << " s ("
              << 100*t_integ/total_time   << "%)\n";
    std::cout << "  Diagnostics/IO:     " << t_diag    << " s ("
              << 100*t_diag/total_time    << "%)\n";
    std::cout << "  Total:              " << total_time << " s\n";
    std::cout << "  Threads used:       " << cfg.n_threads << "\n";
    std::cout << "============================\n\n";

    return total_time;
}

// =============================================================================
// SECTION 8: Verification Tests
// =============================================================================

// --- Test 1: Free Fall ---
// Single particle dropped from rest; compare z(t) with analytical solution
void test_free_fall()
{
    std::cout << "--- Test 1: Free Fall ---\n";

    SimConfig cfg;
    cfg.Lx = 10.0; cfg.Ly = 10.0; cfg.Lz = 10.0;
    cfg.kn = 1e5; cfg.gamma_n = 0.0;
    cfg.gx = 0.0; cfg.gy = 0.0; cfg.gz = -9.81;
    cfg.dt = 1e-4; cfg.t_end = 0.3;
    cfg.out_freq = 100; cfg.out_file = "test_freefall";
    cfg.n_threads = 1; cfg.use_neighbour_search = false;

    std::vector<Particle> parts;
    double z0 = 5.0;
    // Place particle far from walls so no contact occurs
    init_single_particle(parts, 5.0, 5.0, z0, 0,0,0, 1.0, 0.05);

    int total_steps = (int)(cfg.t_end / cfg.dt);
    omp_set_num_threads(1);

    std::ofstream f("test_freefall_traj.csv");
    f << "t,z_num,z_ana,error\n";

    for (int step = 0; step <= total_steps; ++step) {
        double t = step * cfg.dt;
        double z_ana = z0 + 0.5 * cfg.gz * t * t; // z0 - 0.5 g t^2

        if (step % 100 == 0)
            f << t << "," << parts[0].z << "," << z_ana
              << "," << std::abs(parts[0].z - z_ana) << "\n";

        zero_forces(parts);
        add_gravity(parts, cfg);
        // No wall contacts during free fall (particle stays mid-air for t_end=0.3)
        integrate_particles(parts, cfg);
    }

    double z_final_num = parts[0].z;
    double z_final_ana = z0 + 0.5 * cfg.gz * cfg.t_end * cfg.t_end;
    std::cout << "  z_numerical = " << z_final_num
              << "  z_analytical = " << z_final_ana
              << "  error = " << std::abs(z_final_num - z_final_ana) << "\n\n";
}

// --- Test 2: Constant Velocity (zero gravity) ---
void test_constant_velocity()
{
    std::cout << "--- Test 2: Constant Velocity ---\n";

    SimConfig cfg;
    cfg.Lx=10; cfg.Ly=10; cfg.Lz=10;
    cfg.kn=1e5; cfg.gamma_n=0;
    cfg.gx=cfg.gy=cfg.gz=0.0;  // zero gravity
    cfg.dt=1e-4; cfg.t_end=0.5;
    cfg.out_freq=1000; cfg.out_file="test_constvel";
    cfg.n_threads=1; cfg.use_neighbour_search=false;

    double vz0 = 2.0;
    std::vector<Particle> parts;
    init_single_particle(parts, 5,5,5, 0,0,vz0, 1.0, 0.05);

    int total_steps = (int)(cfg.t_end / cfg.dt);
    omp_set_num_threads(1);

    for (int step = 0; step <= total_steps; ++step) {
        zero_forces(parts);
        add_gravity(parts,cfg);
        integrate_particles(parts,cfg);
    }

    double z_expected = 5.0 + vz0 * cfg.t_end;
    std::cout << "  z_numerical = " << parts[0].z
              << "  z_expected  = " << z_expected
              << "  error = " << std::abs(parts[0].z - z_expected) << "\n\n";
}

// --- Test 3: Bouncing Particle ---
// Drop particle onto floor; verify rebound height decreases with damping
void test_bounce()
{
    std::cout << "--- Test 3: Bouncing Particle ---\n";

    SimConfig cfg;
    cfg.Lx=2; cfg.Ly=2; cfg.Lz=5;
    cfg.kn=1e5; cfg.gamma_n=50.0;
    cfg.gx=0; cfg.gy=0; cfg.gz=-9.81;
    cfg.dt=1e-5; cfg.t_end=2.0;
    cfg.out_freq=500; cfg.out_file="test_bounce";
    cfg.n_threads=1; cfg.use_neighbour_search=false;

    std::vector<Particle> parts;
    double z0 = 1.0;
    init_single_particle(parts, 1,1,z0, 0,0,0, 1.0, 0.05);

    int total_steps = (int)(cfg.t_end / cfg.dt);
    omp_set_num_threads(1);

    std::ofstream f("test_bounce_traj.csv");
    f << "t,z,vz\n";

    double prev_vz = 0, prev_z = z0;
    int bounce_count = 0;

    for (int step = 0; step <= total_steps; ++step) {
        double t = step * cfg.dt;
        if (step % 100 == 0)
            f << t << "," << parts[0].z << "," << parts[0].vz << "\n";

        // Detect bounce (velocity sign change from - to +)
        if (step > 0 && prev_vz < 0 && parts[0].vz > 0) {
            ++bounce_count;
            std::cout << "  Bounce " << bounce_count
                      << " at t=" << t
                      << "  z=" << parts[0].z << "\n";
        }
        prev_vz = parts[0].vz;
        prev_z  = parts[0].z;

        zero_forces(parts);
        add_gravity(parts, cfg);
        compute_wall_contacts(parts, cfg);
        integrate_particles(parts, cfg);
    }
    std::cout << "  Total bounces detected: " << bounce_count << "\n\n";
}

// =============================================================================
// SECTION 9: Scaling / Performance Study
// =============================================================================

// Runs the simulation silently and returns wall-clock time
// Used to build speedup tables
double timed_run(int N, int n_threads, double t_end, bool use_ns)
{
    SimConfig cfg;
    cfg.Lx=2.0; cfg.Ly=2.0; cfg.Lz=2.0;
    cfg.kn=1e5; cfg.gamma_n=50.0;
    cfg.gx=0; cfg.gy=0; cfg.gz=-9.81;
    cfg.dt=1e-4; cfg.t_end=t_end;
    cfg.out_freq=99999; cfg.out_file="scaling_tmp";
    cfg.n_threads=n_threads;
    cfg.use_neighbour_search=use_ns;

    std::vector<Particle> parts;
    init_random_particles(parts, N, cfg, 0.04, 1.0);

    return run_simulation(parts, cfg, false);
}

void scaling_study()
{
    std::cout << "\n===== Strong Scaling Study =====\n";
    std::cout << std::left << std::setw(12) << "Threads"
              << std::setw(12) << "Time(s)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Efficiency\n";
    std::cout << std::string(48, '-') << "\n";

    int N = 1000;
    double t_end = 0.05;

    double T1 = timed_run(N, 1, t_end, false);
    std::vector<int> thread_counts = {1, 2, 4, 8};

    std::ofstream sf("scaling_results.csv");
    sf << "threads,time,speedup,efficiency\n";

    for (int np : thread_counts) {
        double Tp = timed_run(N, np, t_end, false);
        double S  = T1 / Tp;
        double E  = S / np;
        std::cout << std::setw(12) << np
                  << std::setw(12) << std::fixed << std::setprecision(3) << Tp
                  << std::setw(12) << S
                  << std::setw(12) << E << "\n";
        sf << np << "," << Tp << "," << S << "," << E << "\n";
    }
    std::cout << "\n";
}

// =============================================================================
// SECTION 10: main()
// =============================================================================

int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------
    // 10.1  Verification Tests
    // ------------------------------------------------------------------
    std::cout << "============================================\n";
    std::cout << " DEM Solver – Verification Tests\n";
    std::cout << "============================================\n\n";

    test_free_fall();
    test_constant_velocity();
    test_bounce();

    // ------------------------------------------------------------------
    // 10.2  Multi-particle Simulations
    // ------------------------------------------------------------------
    std::cout << "============================================\n";
    std::cout << " Multi-Particle Simulations\n";
    std::cout << "============================================\n\n";

    // Particle counts to investigate
    std::vector<int> N_list = {200, 500, 1000};

    for (int N : N_list) {
        std::cout << "--- N = " << N << " particles ---\n";

        SimConfig cfg;
        cfg.Lx = 2.0; cfg.Ly = 2.0; cfg.Lz = 2.0;
        cfg.kn = 1e5; cfg.gamma_n = 50.0;
        cfg.gx = 0.0; cfg.gy = 0.0; cfg.gz = -9.81;
        cfg.dt  = 1e-4;
        cfg.t_end = 0.5;
        cfg.out_freq = 500;
        cfg.out_file = "sim_N" + std::to_string(N);
        cfg.n_threads = 2;
        cfg.use_neighbour_search = false; // use neighbour search for large N

        std::vector<Particle> parts;
        init_random_particles(parts, N, cfg, 0.04, 1.0);

        run_simulation(parts, cfg, true);
    }

    // ------------------------------------------------------------------
    // 10.3  Scaling Study (OpenMP speedup)
    // ------------------------------------------------------------------
    scaling_study();

    std::cout << "All runs complete. Check CSV output files.\n";
    return 0;
}