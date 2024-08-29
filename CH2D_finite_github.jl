# File Name: CH2D_finite_github.jl
# Author: Edgar Avalos
# Last Modified: 2024-08-29
# Version: 1.0
#
# Description:
# This script solves Cahn-Hilliard coupled equations in 2D.
#
# Dependencies:
# - Random
# - Plots
# - SparseArrays
# - LinearAlgebra
#

using Random
using Plots
using SparseArrays
using LinearAlgebra


# Define the build_laplacian function
function build_laplacian(N, DX)
    L = spzeros(N^2, N^2)
    for i in 1:N
        for j in 1:N
            idx = (i - 1) * N + j
            L[idx, idx] = -4
            L[idx, (mod1(i + 1, N) - 1) * N + j] = 1
            L[idx, (mod1(i - 1, N) - 1) * N + j] = 1
            L[idx, (i - 1) * N + mod1(j + 1, N)] = 1
            L[idx, (i - 1) * N + mod1(j - 1, N)] = 1
        end
    end
    return L / DX^2
end



# Initialization function
function initialize_array(N, value, noise)
    return value .+ noise * rand(N, N)
end


function solve_pde(n, nsteps, dt, eps, epsv, sigma, a, b1, b2, tauv, u0, v0, L)
    # Precompute constant matrices
    LL = L * L
    M1 = I + dt * eps^2 * LL + dt * (1 - a) * L
    M2 = tauv * I + dt * epsv^2 * LL + dt * (1 - a) * L

    # Convert M1 and M2 to sparse format
    M1 = sparse(M1)
    M2 = sparse(M2)

    # Precompute LU decompositions
    lu_M1 = lu(M1)
    lu_M2 = lu(M2)

    u = u0
    v = v0

    for itt in 1:nsteps
        #println("Step $(itt)/$(nsteps)")

        # terms that depend on the current u and v
        u_cubed = u .^ 3
        v_cubed = v .^ 3
        v_squared = v .^ 2

        BB1 = u + dt * L * (u_cubed - a * u + b1 * v - 0.5 * b2 * v_squared)
        BB2 = tauv * v + dt * L * (v_cubed - a * v + b1 * u - b2 * u .* v) - (dt * sigma) * v

        # LU factors to solve the linear systems
        u = lu_M1 \ BB1 #  M1 *u =BB1 -> (L U) u=BB1  -> L y = BB1, U * u = y
        v = lu_M2 \ BB2
    end

    return u, v
end

# visualize two arrays side by side
function visualize_arrays(u, v, title_u="Array u", title_v="Array v")
    p1 = heatmap(u, title=title_u, aspect_ratio=:equal, color=:coolwarm)
    p2 = heatmap(v, title=title_v, aspect_ratio=:equal, color=:viridis)
    plot_obj = plot(p1, p2, layout=(1, 2), size=(800, 400))
    display(plot_obj)  # Ensure the plot is displayed
end


# Define the main function
function main()
    N_STEPS = 500
    DT = 0.002
    N = 50
    DX = 0.016
    #L = N * DX
    NOISE = 0.4
    U0 = -0.4
    V0 = -0.2
    EPSILON2 = 0.04 #0.0025
    SIGMA = 50.0
    B1 = 0.0
    B2 = 1.0
    TV = 1.0
    A=3.0
    SEED = 1231

    #fix the random
    Random.seed!(SEED)
    
    # Initialize arrays
    u = initialize_array(N, U0, NOISE)
    v = initialize_array(N, V0, NOISE)

    # Visualize the initial arrays
    #visualize_arrays(u, v, "Initial Array u", "Initial Array v")

    L = build_laplacian(N, DX)

    println("Size of L: ", size(L))

    # Reshape u and v into column vectors
    u_col = reshape(u, :, 1)
    v_col = reshape(v, :, 1)
    println("Size of u_col: ", size(u_col))
    println("Size of v_col: ", size(v_col))

    # Update arrays u and v
    u, v = solve_pde(N, N_STEPS, DT, EPSILON2, EPSILON2, SIGMA, A, B1, B2, TV, u_col, v_col, L)


    # Visualize the updated arrays
    #reshape u and v into 2D arrays
    u = reshape(u, N, N)
    v = reshape(v, N, N)
    visualize_arrays(u, v, "Updated Array u", "Updated Array v")
end

@time main()
