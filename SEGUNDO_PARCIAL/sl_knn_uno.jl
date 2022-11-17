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

gr(size = (600, 600,600))

# generate random points for reference

Random.seed!(1)

f1_train = rand(100)

f2_train = rand(100)

f3_train = rand(100)

p_knn = scatter(f1_train, f2_train,f3_train,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    zlabel = "Feature 3",
    title = "k-NN & k-D Tree Demo",
    legend = false,
    color = :blue
)

# build tree (SUBarte 2)

X_train = [f1_train f2_train f3_train]
#Transpuesta NearestNeighbors requieres que este en horizontal en vez de vertical
X_train_t = permutedims(X_train)

kdtree = KDTree(X_train_t)

# initialize k for k-NN

k = 11

# generate random point for testing de K-NN

f1_test = rand()

f2_test = rand()

f3_test = rand()

X_test = [f1_test, f2_test, f3_test]

# add test point to plot

scatter!([f1_test], [f2_test], [f3_test],
    color = :red, markersize = 10
)

# find nearest neighbors using k-NN & k-d tree
#Ordena la salida por distancia 
#SUBarte 3
index_knn, distances = knn(kdtree, X_test, k, true)

# display output

output = [index_knn distances]

vscodedisplay(output)

#SUBarte 4
# plot nearest neighbors
#Genera las coordenadas de los (x) 11 vecinos mas cercanos
f1_knn = [f1_train[i] for i in index_knn]
#Genera las coordenadas de los (y) 11 vecinos mas cercanos
f2_knn = [f2_train[i] for i in index_knn]

f3_knn = [f3_train[i] for i in index_knn]

#Graficamos los 11 vecinos mas  (puntos amarillos)
scatter!(f1_knn, f2_knn, f3_knn,
    color = :yellow, markersize = 10, alpha = 0.5
)

# connect test point with nearest neighbors
#plot!([f1_test, f1_knn[i]], [f2_test, f2_knn[i]] , (X,Y)
for i in 1:k
    plot!([f1_test, f1_knn[i]], [f2_test, f2_knn[i]],[f3_test, f3_knn[i]],
        color = :green
    )
end

p_knn #SCATTER NECESITA SER LLAMADO DE NUEVO PARA VER LAS

# save plot

savefig(p_knn, "knn_concept_plot_uno.svg")
