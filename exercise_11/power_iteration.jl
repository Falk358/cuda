# Define functions, constants, and types
using CUDA
using LinearAlgebra

# computes normed matrix multiplication
function cuda_norm!(vec, res)
    i = threadIdx().x
    @inbounds res[1] += vec[i]^2

    res[1] = sqrt(res[1])

    return nothing
end

function powerIteration(mat, max_iter::Int64)
    vec_size = size(mat,2) # size of 2nd dimension of given matrix
    vec = CUDA.rand(vec_size)
    for i in 1:max_iter
        vec_buf = mat * vec
        vec_norm = CuArray([0.0])
        @cuda threads = 8 cuda_norm!(vec_buf, vec_norm) # start cuda kernel with 8 threads, saves result in vec_norm
        if i == max_iter
            println(" Maximal eigenvalue of matrix computed by cuda norm is: $vec_norm")
        end
        vec = vec_buf ./ vec_norm 
    end
    return vec
end

function main()
    h_matrix = [[1, 2, 3, 4] [5, 6, 7, 8] [9, 10, 11, 12] [13, 14, 15, 16]]
    d_matrix = CuArray(h_matrix)
    correct_solution = eigen(h_matrix)
    estimated_solution =powerIteration(d_matrix, 1000)
    println("The eigenvalues of our given matrix computed with LinearAlgebra package are:")
    for value in correct_solution.values
        println("$value")
    end
    println("The eigenvectors of our given matrix computed with LinearAlgebra package are:")
    for vector in correct_solution.vectors
        println("$vector")
    end
    println("Eigenvector associated with biggest eigenvalue estimated by power Iteration is: $estimated_solution")
end

# Check if the script is being run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()  # Call the main function if the script is run directly
end
