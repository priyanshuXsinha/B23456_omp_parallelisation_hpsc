#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <algorithm>

// Data structure for a particle
struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
    double mass, radius;
};

// --- Simulation Parameters ---
const int N = 1000;            // Number of particles
const double dt = 1e-4;        // Timestep
const double t_max = 2.0;      // Total simulation time
const double g = 9.81;         // Gravity
const double kn = 1e5;         // Normal stiffness
const double gamma_n = 50.0;   // Damping coefficient
const double Lx = 0.5, Ly = 0.5, Lz = 1.0; // Box dimensions

// Function Prototypes
void initialize_particles(std::vector<Particle>& p);
void zero_forces(std::vector<Particle>& p);
void add_gravity(std::vector<Particle>& p);
void compute_particle_contacts(std::vector<Particle>& p);
void compute_wall_contacts(std::vector<Particle>& p);
void integrate_particles(std::vector<Particle>& p);
double compute_kinetic_energy(const std::vector<Particle>& p);
void write_output(const std::vector<Particle>& p, int step);

int main() {
    std::vector<Particle> particles(N);
    initialize_particles(particles);

    int num_steps = t_max / dt;
    int output_freq = 100; // Output every 100 steps

    double start_time = omp_get_wtime();

    // Main Simulation Loop
    for (int step = 0; step < num_steps; ++step) {
        zero_forces(particles);
        add_gravity(particles);
        compute_particle_contacts(particles);
        compute_wall_contacts(particles);
        integrate_particles(particles);

        if (step % output_freq == 0) {
            double ke = compute_kinetic_energy(particles);
            std::cout << "Step: " << step << " | Time: " << step * dt 
                      << " | KE: " << ke << std::endl;
            // write_output(particles, step); // Uncomment to save data
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "Simulation complete. Total runtime: " << (end_time - start_time) << " seconds.\n";

    return 0;
}

void initialize_particles(std::vector<Particle>& p) {
    // Simple grid initialization to prevent initial overlaps
    int particles_per_side = std::ceil(std::cbrt(N));
    double spacing = 0.04;
    int count = 0;

    for (int ix = 0; ix < particles_per_side && count < N; ++ix) {
        for (int iy = 0; iy < particles_per_side && count < N; ++iy) {
            for (int iz = 0; iz < particles_per_side && count < N; ++iz) {
                p[count].x = 0.1 + ix * spacing;
                p[count].y = 0.1 + iy * spacing;
                p[count].z = 0.1 + iz * spacing;
                p[count].vx = 0.0; p[count].vy = 0.0; p[count].vz = 0.0;
                p[count].mass = 0.01;
                p[count].radius = 0.015;
                count++;
            }
        }
    }
}

void zero_forces(std::vector<Particle>& p) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        p[i].fx = 0.0; p[i].fy = 0.0; p[i].fz = 0.0;
    }
}

void add_gravity(std::vector<Particle>& p) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        p[i].fz -= p[i].mass * g; // Gravity acts in -z direction
    }
}

void compute_particle_contacts(std::vector<Particle>& p) {
    // OpenMP Parallelization of O(N^2) loop
    // Using schedule(dynamic) helps balance the load, and atomic prevents race conditions
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double rx = p[j].x - p[i].x;
            double ry = p[j].y - p[i].y;
            double rz = p[j].z - p[i].z;
            
            double d_sq = rx*rx + ry*ry + rz*rz;
            double r_sum = p[i].radius + p[j].radius;

            // Broad-phase check using squared distance (faster)
            if (d_sq < r_sum * r_sum) {
                double d = std::sqrt(d_sq);
                double delta = r_sum - d;
                
                if (delta > 0 && d > 1e-12) {
                    double nx = rx / d;
                    double ny = ry / d;
                    double nz = rz / d;

                    double vn_x = p[j].vx - p[i].vx;
                    double vn_y = p[j].vy - p[i].vy;
                    double vn_z = p[j].vz - p[i].vz;
                    
                    double vn = vn_x * nx + vn_y * ny + vn_z * nz;

                    // Spring-dashpot model: Fn = max(0, kn*delta - gamma_n*vn)
                    double Fn = std::max(0.0, kn * delta - gamma_n * vn);

                    double fx = Fn * nx;
                    double fy = Fn * ny;
                    double fz = Fn * nz;

                    // Atomic updates are critical here to prevent race conditions 
                    // when multiple threads try to update the same particle simultaneously
                    #pragma omp atomic
                    p[i].fx -= fx;
                    #pragma omp atomic
                    p[i].fy -= fy;
                    #pragma omp atomic
                    p[i].fz -= fz;

                    #pragma omp atomic
                    p[j].fx += fx;
                    #pragma omp atomic
                    p[j].fy += fy;
                    #pragma omp atomic
                    p[j].fz += fz;
                }
            }
        }
    }
}

void compute_wall_contacts(std::vector<Particle>& p) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double R = p[i].radius;
        
        // Z walls (Floor and Ceiling)
        if (p[i].z - R < 0) { // Bottom wall
            double delta = R - p[i].z;
            double vn = -p[i].vz; 
            double Fn = std::max(0.0, kn * delta - gamma_n * vn);
            p[i].fz += Fn;
        } else if (p[i].z + R > Lz) { // Top wall
            double delta = (p[i].z + R) - Lz;
            double vn = p[i].vz;
            double Fn = std::max(0.0, kn * delta - gamma_n * vn);
            p[i].fz -= Fn;
        }

        // X walls
        if (p[i].x - R < 0) {
            double delta = R - p[i].x;
            double vn = -p[i].vx;
            double Fn = std::max(0.0, kn * delta - gamma_n * vn);
            p[i].fx += Fn;
        } else if (p[i].x + R > Lx) {
            double delta = (p[i].x + R) - Lx;
            double vn = p[i].vx;
            double Fn = std::max(0.0, kn * delta - gamma_n * vn);
            p[i].fx -= Fn;
        }

        // Y walls
        if (p[i].y - R < 0) {
            double delta = R - p[i].y;
            double vn = -p[i].vy;
            double Fn = std::max(0.0, kn * delta - gamma_n * vn);
            p[i].fy += Fn;
        } else if (p[i].y + R > Ly) {
            double delta = (p[i].y + R) - Ly;
            double vn = p[i].vy;
            double Fn = std::max(0.0, kn * delta - gamma_n * vn);
            p[i].fy -= Fn;
        }
    }
}

void integrate_particles(std::vector<Particle>& p) {
    // Semi-implicit Euler
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        // Update velocity first
        p[i].vx += (p[i].fx / p[i].mass) * dt;
        p[i].vy += (p[i].fy / p[i].mass) * dt;
        p[i].vz += (p[i].fz / p[i].mass) * dt;

        // Update position using new velocity
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

double compute_kinetic_energy(const std::vector<Particle>& p) {
    double total_ke = 0.0;
    #pragma omp parallel for reduction(+:total_ke)
    for (int i = 0; i < N; ++i) {
        double v_sq = p[i].vx*p[i].vx + p[i].vy*p[i].vy + p[i].vz*p[i].vz;
        total_ke += 0.5 * p[i].mass * v_sq;
    }
    return total_ke;
}

void write_output(const std::vector<Particle>& p, int step) {
    // Basic CSV output for visualization in tools like Python/ParaView
    std::ofstream file("output_" + std::to_string(step) + ".csv");
    file << "x,y,z,vx,vy,vz,radius\n";
    for (const auto& part : p) {
        file << part.x << "," << part.y << "," << part.z << ","
             << part.vx << "," << part.vy << "," << part.vz << ","
             << part.radius << "\n";
    }
    file.close();
}