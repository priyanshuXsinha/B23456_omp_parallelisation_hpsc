#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>

using namespace std;

// ---------------- Particle संरचना ----------------
struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
    double mass, radius;
};

// ---------------- Global Parameters ----------------
const int N = 500;             // number of particles
const double dt = 1e-4;
const double totalTime = 1.0;
const double kn = 1e5;         // stiffness
const double gamma_n = 10.0;   // damping
const double g = -9.81;

// Box size
const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

// ---------------- Initialize ----------------
void initialize(vector<Particle> &p) {
    for (int i = 0; i < N; i++) {
        p[i].x = drand48() * Lx;
        p[i].y = drand48() * Ly;
        p[i].z = drand48() * Lz;

        p[i].vx = p[i].vy = p[i].vz = 0.0;
        p[i].mass = 1.0;
        p[i].radius = 0.01;
    }
}

// ---------------- Zero Forces ----------------
void zero_forces(vector<Particle> &p) {
    for (int i = 0; i < N; i++) {
        p[i].fx = p[i].fy = p[i].fz = 0.0;
    }
}

// ---------------- Gravity ----------------
void apply_gravity(vector<Particle> &p) {
    for (int i = 0; i < N; i++) {
        p[i].fz += p[i].mass * g;
    }
}

// ---------------- Particle Contact ----------------
void compute_contacts(vector<Particle> &p) {

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            double rx = p[j].x - p[i].x;
            double ry = p[j].y - p[i].y;
            double rz = p[j].z - p[i].z;

            double dist = sqrt(rx*rx + ry*ry + rz*rz);
            double overlap = p[i].radius + p[j].radius - dist;

            if (overlap > 0) {

                double nx = rx / dist;
                double ny = ry / dist;
                double nz = rz / dist;

                double dvx = p[j].vx - p[i].vx;
                double dvy = p[j].vy - p[i].vy;
                double dvz = p[j].vz - p[i].vz;

                double vn = dvx*nx + dvy*ny + dvz*nz;

                double fn = max(0.0, kn * overlap - gamma_n * vn);

                double fx = fn * nx;
                double fy = fn * ny;
                double fz = fn * nz;

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

// ---------------- Wall Contact ----------------
void compute_wall_contacts(vector<Particle> &p) {

    for (int i = 0; i < N; i++) {

        // Z bottom
        double overlap = p[i].radius - p[i].z;
        if (overlap > 0) {
            double vn = p[i].vz;
            double fn = max(0.0, kn * overlap - gamma_n * vn);
            p[i].fz += fn;
        }

        // Z top
        overlap = p[i].z + p[i].radius - Lz;
        if (overlap > 0) {
            double vn = p[i].vz;
            double fn = max(0.0, kn * overlap + gamma_n * vn);
            p[i].fz -= fn;
        }

        // X walls
        overlap = p[i].radius - p[i].x;
        if (overlap > 0) p[i].fx += kn * overlap;

        overlap = p[i].x + p[i].radius - Lx;
        if (overlap > 0) p[i].fx -= kn * overlap;

        // Y walls
        overlap = p[i].radius - p[i].y;
        if (overlap > 0) p[i].fy += kn * overlap;

        overlap = p[i].y + p[i].radius - Ly;
        if (overlap > 0) p[i].fy -= kn * overlap;
    }
}

// ---------------- Integration ----------------
void integrate(vector<Particle> &p) {

    for (int i = 0; i < N; i++) {

        // velocity update
        p[i].vx += (p[i].fx / p[i].mass) * dt;
        p[i].vy += (p[i].fy / p[i].mass) * dt;
        p[i].vz += (p[i].fz / p[i].mass) * dt;

        // position update
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// ---------------- Energy ----------------
double compute_energy(vector<Particle> &p) {
    double ke = 0.0;
    for (int i = 0; i < N; i++) {
        ke += 0.5 * p[i].mass *
              (p[i].vx*p[i].vx +
               p[i].vy*p[i].vy +
               p[i].vz*p[i].vz);
    }
    return ke;
}

// ---------------- Main ----------------
int main() {

    vector<Particle> particles(N);

    initialize(particles);

    int steps = totalTime / dt;

    ofstream file("energy.dat");

    for (int t = 0; t < steps; t++) {

        zero_forces(particles);
        apply_gravity(particles);
        compute_contacts(particles);
        compute_wall_contacts(particles);
        integrate(particles);

        double ke = compute_energy(particles);

        file << t * dt << " " << ke << endl;

        if (t % 100 == 0)
            cout << "Step: " << t << " KE: " << ke << endl;
    }

    file.close();
    return 0;
}