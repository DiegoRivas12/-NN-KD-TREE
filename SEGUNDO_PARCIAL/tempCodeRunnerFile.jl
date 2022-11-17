################################################################################
# ML > Supervised Learning > k-Nearest Neighbors
################################################################################

################################################################################
# Part 1: Concepts
################################################################################

# load packages

using NearestNeighbors, Plots

using Random

# initialize plot

gr(size = (600, 600))

# generate random points for reference

Random.seed!(1)

f1_train = rand(100)

f2_train = rand(100)

p_knn = scatter(f1_train, f2_train,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "k-NN & k-D Tree Demo",
    legend = false,
    color = :blue
)