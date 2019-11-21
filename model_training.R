require(bmp)
require(OpenImageR)
require(stringr)

# Divide into train and test set using K-fold -------------------
data.Partition <- function(data,data.names,k = 6, n.train = 5) {
  Partition = NULL
  
  matrix_ind = matrix(rep(0,480),nrow = 80,ncol = 6)
  colnames(matrix_ind) <- c("Face1","Face2","Face3","Face4","Face5","Face6")
  rownames(matrix_ind) <- unique(str_split_fixed(data.names,'-',n=2)[,1])
  for (i in 1:nrow(matrix_ind)){
    index = c(sample(1:6,6) + 6*(i-1))
    matrix_ind[i,] <- index 
  }
  
  Partition$train = data[-matrix_ind[,k],]
  Partition$test = data[matrix_ind[,k],]
  Partition$indices.train = rep(rownames(matrix_ind),each = n.train)
  Partition$indices.test = names(matrix_ind[,k])
  return(Partition)
}

#PCA ---------------
PCA <- function(data_train,var.explained = 0.95){
  PCA = NULL
  #We scale train: 
  PCA$scaled_train = scale(data_train, center = T, scale = T)
  #Compute the convariance matrix
  Sigma_ = (PCA$scaled_train) %*% t(PCA$scaled_train)/(nrow(PCA$scaled_train)-1)
  #Get it's eigenvectors and eigenvalues
  Eigen = eigen(Sigma_)
  PCA$Eigenvalues = Eigen$values
  Prop.Var = PCA$Eigenvalues/sum(PCA$Eigenvalues)
  Cummulative.Var = cumsum(PCA$Eigenvalues)/sum(PCA$Eigenvalues)
  
  PCA$n.eigenVectors <- min(which(Cummulative.Var>var.explained))
  PCA$Eigenvectors = Eigen$vectors[,1:PCA$n.eigenVectors]
  PCA$Eigenfaces = t(PCA$scaled_train)%*%PCA$Eigenvectors
  
  return(PCA)
}

getDistances<- function(w.train,w.test,distance = 'Euclidean'){
  dists = rep(0,nrow(w.train))
  if(distance == "Euclidean"){
    dists = sqrt(rowSums((sweep(w.train,2,w.test))^2))
    return(dists)
  }else if(distance == "Manhattan"){
    dists = rowSums(abs(sweep(w.train,2,w.test)))/(rowSums(abs(w.train))*sum(abs(w.test)))
    return(dists)
  }else if(distance == "Maximum"){
    dists = apply(abs(sweep(w.train,2,w.test)),1,max)
    return(dists)
  }else if(distance == "MSSE"){
    dists = rowSums((sweep(w.train,2,w.test))^2)/(rowSums((w.train)^2)*sum((w.test)^2))
    return(dists)
  }else{
    warning(paste(distance,"is not an compatible distance measure."))
    return(rep(NA,ncol(w.train)))
  }
}

#MAIN -----------
#Set working directory:
this_file <- rstudioapi::getSourceEditorContext()$path
this_dir <- dirname(this_file)
setwd(paste(this_dir,'/Faces', sep = ''))

#Get data from the Train/Faces directory: 
names = list.files(pattern = "bmp") ##Assign the labels to each face
##to get a matrix of the 480 faces (each face is a vector):
data = matrix(0, length(names),165*120*3)
for (i in 1:length(names)){
  Im = read.bmp(names[i])
  red = as.vector(Im[,,1]) #each vector is of size 120*165 px = 19800
  green = as.vector(Im[,,2])
  blue = as.vector(Im[,,3])
  data[i,] = t(c(red, green, blue))
}

##Average face of all the data:
data.mean = colMeans(data)
avg_Im = array( 255*(data.mean - min(data.mean))/(max(data.mean)-min(data.mean)) ,dim(Im))
#imageShow(avg_Im)

#TEST THE MODEL --------------------------------
my.distances = c("Euclidean","Maximum","MSSE","Manhattan")#Distance measures to test the model with
dist_testing.df = data.frame(matrix(rep(0,length(my.distances)*6),nrow = 6,ncol = 4))
rownames(dist_testing.df) <- c("k=1","k=2","k=3","k=4","k=5","k=6")
colnames(dist_testing.df) <- my.distances
manhattanDists.df = data.frame(manhattan.max = 1:6,
                               manhattan.mean = 1:6) 
classification = max.dists = mean.dists = rep(0,80)

#To get al the information about the test for each partition 
library(tictoc)
{
  tic()
  for(k in 1:6){
    #Paritition the data into train and test sets: 
    partition = data.Partition(data = data,names,k = k)
    data_train = partition$train #train has 400 photographies
    data_test = partition$test #test has 80 photographies
    #Perform pca on the training set
    PCA.train = PCA(data_train)
    PCA.train$n.eigenVectors #Is the number of principal components that explain ~95% of variability
    #Scale train and test sets:
    scaled_train = PCA.train$scaled_train
    scaled_test = scale(data_test,
                        center = attr(scaled_train,"scaled:center"),
                        scale = attr(scaled_train,"scaled:scale"))
    #Project the training and test sets on the Eigenspace of n principal components
    w.train = t(t(PCA.train$Eigenfaces)%*%t(scaled_train)) 
    w.test = t(t(PCA.train$Eigenfaces)%*%t(scaled_test))
    
    kdistances.ccr = 1:length(my.distances)
    for(d in 1:length(my.distances)){
      distance = my.distances[d]
      for(i in 1:nrow(data_test)){
        #Img = array(scaled_test[i,],c(165,120,3)) 
        #imageShow(Img)
        
        dists = getDistances(w.train,w.test[i,],distance)
        max.dists[i] = max(dists)
        mean.dists[i] = mean(dists)
        #We classify the image based on the firt images with the less distance    
        closest = partition$indices.train[order(dists)][1:5]
        #We make a table of the number of ocurrences of each label (person)
        closest.t <- sort(table(closest), decreasing = T)
        table.names <- names(closest.t) 
        #We classify the label with the most ocurrences within the first 5
        to.classify <- table.names[which(closest.t == max(closest.t))]
        if(length(to.classify)>1){
          #If there's 2 or more labels with the same number of ocurrences then we grab the onet that is 
          #closest to the test image (the one that appears first in the vector with the closest distances)
          classification[i] = closest[min(match(to.classify,closest))]
        }else{
          #Else we grab the label with the highest amount of ocurrences
          classification[i] = table.names[which(closest.t == max(closest.t))][1]
        }
      }
      ccr.model = sum(partition$indices.test == classification)/nrow(data_test)
      kdistances.ccr[d] = ccr.model
      if (distance == "Manhattan"){
        manhattanDists.df[k,] = c(max(max.dists),mean(mean.dists))
      }
    }
    dist_testing.df[k,] = kdistances.ccr
  }
  dist_testing.df <- cbind(dist_testing.df,manhattanDists.df)
  dist_testing.df <- rbind(dist_testing.df, avg = round(colMeans(dist_testing.df),4))
  dist_testing.df
  toc()
}

#Classification data obtained using the Manhattan Distance
classification.data <- data.frame(true.names = as.character(partition$indices.test),
                                  classified.as = as.character(classification),
                                  max.dists)
classification.data
ccr.model = sum(partition$indices.test == classification)/nrow(classification.data)
ccr.model
