
#PCA-BASED FACE CLASSIFIER

#FUNCTION: 
#- Receives a matrix of dimensions nx59400 representing n photographs of size 120x165 px each.
#- Returns a vectors of n labels of values 0-80 where 0 means that the person corresponding to the 
#photograph was not part of the train set and 1-80, the label/name of the person recognized.
classify.faces <- function(test, #Test matrix is the only "real" input
                           Eigenfaces,w.train,labels.train, train_scaled.attr){
    
    threshold = 3.3e-06 #To classify impostors
    test.scaled = scale(test,
                        center = train_scaled.attr$center,
                        scale = train_scaled.attr$scale)
    
    #Project the test set on the Eigenspace of n principal components
    w.test = t(t(Eigenfaces)%*%t(test.scaled))
    
    classification = rep(0,nrow(test))
    for(i in 1:nrow(test)){ #For each row/photograph of the test matrix
        
        #We get the Manhattan distance of the representation of the photograph 
        #and each of the original photographs in the train set.  
        dists = rowSums(abs(sweep(w.train,2,w.test[i,])))/(rowSums(abs(w.train))*sum(abs(w.test[i,])))
        
        if(min(dists)>threshold){ #The face might be an impostor 
            classification[i] = 0
        }else{ #The face might not be an impostor 
            
            #We classify the image based on the firt 6 images with the less distance    
            closest = labels.train[order(dists)][1:6]
            #We make a table of the number of ocurrences of each label (person in train set)
            closest.t <- sort(table(closest), decreasing = T)
            table.names <- names(closest.t) 
            #We classify the label was the one with the most ocurrences within the first 5
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
    }
    #Return the vector of classifications
    return(classification)
}

#MAIN ---------------------------------------------------

#Set working directory first to the one where the data files are in:
this_dir <- dirname(rstudioapi::getSourceEditorContext()$path); #Only works with Rstudio
setwd(this_dir)

#Then load the necessary data from that directory
load("indices_train.Rdata") #is a vector with 480 labels (1-80) identifying each picture in the given train set.  
load("train_scaled.attr.Rdata") #The center and variance of the original train set to scale the test based on it. 
load("Eigenfaces.Rdata") #The eigenfaces matrix of size 156x59400
load("w_train.Rdata") #Projection of the train test onto the eigenspace of n principal component

# Get test data
setwd(choose.dir()) #<- Directory where the test photographies are in.
test.names = list.files(pattern = "bmp") #Read bmp files in that directory
test.m = matrix(0, length(test.names),165*120*3)#<- matrix of size nx59400 where n = number of photographies in directory
for (i in 1:length(test.names)){
    Im = read.bmp(test.names[i])
    red = as.vector(Im[,,1]) #each vector is of size 120*165 px = 19800
    green = as.vector(Im[,,2])
    blue = as.vector(Im[,,3])
    test.m[i,] = t(c(red, green, blue))
}

classified.labels = classify.faces(test.m, #<- test matrix from above
                                     Eigenfaces,w.train,indices.train,train_scaled.attr)#<- Loaded objects
classified.labels #<- Vector of n labels corresponding to the classifications generated by the function.  
ccr <- sum(classified.labels == indices.test)/length(test.names) #To get model Accuracy
names(ccr) <- "Manhattan"
