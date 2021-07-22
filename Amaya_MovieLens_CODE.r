##############################################################
##          MACHINE LEARNING MODEL FOR PREDICTION           ##
##       OF MOVIE RATINGS IN AN ONLINE MEDIA SERVICE        ##
##         USING THE MATRIX FACTORIZATION METHOD            ##
##############################################################
### STAGE 0: DATA GATHERING EXPLORATION AND PREPARATION   ####
##############################################################

# Step 0.1. ----
# Create edx set, validation set (final hold-out test set)


# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           # title = as.character(title),
                                           # genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, 
                                  times = 1, p = 0.1, 
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##############################################################
# Step 0.3 ----
# 'edx' splitting into training and test sets
set.seed(1, sample.kind="Rounding")
movie_test_index <- createDataPartition(y = edx$rating, 
                                        times = 1, p = 0.2, 
                                        list = FALSE)
movieTrain <- edx[-movie_test_index,]
temp1 <- edx[movie_test_index,]

  # Make sure userId and movieId in test set are also in train set
movieTest <- temp1 %>% 
  semi_join(movieTrain, by = "movieId") %>%
  semi_join(movieTrain, by = "userId")

  # Add rows removed from test set back into train set
removed1 <- anti_join(temp1, movieTest)
movieTrain <- rbind(movieTrain, removed1)

  # Remove unnecessary variables
rm(movie_test_index, removed1, temp1)

  # Define function for the later calculation of Residual Mean Squared Error
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }


##############################################################
## RECOMENDATION SYSTEM: Matrix Factorization Using Genres  ##
##############################################################
# STAGE 1: CALCULATE MOVIE BIAS ----
##############################################################

# Step 1.1 ---- 
# General ratings mean
mu <- mean(edx$rating)
mu

# Step 1.2 ----
# Calculate how biased is each movie from the average
movie_bias <- movieTrain %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))


##############################################################
# STAGE 2: CALCULATE USER GENERAL BIAS                     ####
##############################################################

TrMb<- movieTrain %>% 
  left_join(movie_bias, by='movieId') 

user_gb <- TrMb %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))



##############################################################
# STAGE 3: CALCULATE USER BIAS REGARDING EVERY GENRE      ####
##############################################################
  
# Step 3.1. ----
# Which are the genres? // (over the whole 'edx' Dataset), bacause 
                          #// 'movieTest' can include more genres than 
                          #// those in movieTrain.
raw_genres <- strsplit(edx$genres,"\\|")
genres <- unique(unlist(raw_genres))
genres    #// 20 genres including the absence of listed genres


# Step 3.2. ----
# Simplify 'edx'. Group by movieId and create a list
# with the movie genres of unique movies 
sub_train <- data.frame(movieId = movieTrain$movieId, 
                        genres = movieTrain$genres)
gen <- sub_train %>% group_by(movieId) %>% 
  summarize(k=unique(genres))

# Step 3.3. ----
# Build the Matrix 'A', which relates each movie with its genres
A <- matrix(NA,length(gen$movieId), length(genres))

for (i in 1:length(gen$k)){   
  A[i,] <- str_detect(gen$k[i],genres)
}

colnames(A) <- paste("g",1:20, sep="")
rownames(A) <- gen$movieId
is.na(A) <- A == FALSE
head(A)

# Step 3.4. ----
# Build Table 'x'. Build a unique 'users' list
x <- select(movieTrain,userId,movieId,rating)
users <- movieTrain %>% group_by(userId) %>% 
                  summarize(unique(userId))

movies <- movieTrain %>% group_by(movieId) %>% 
  summarize(unique(movieId))


# Step 3.5. ----
# Calculate 'b_ug', user bias for every genre, by entry
  # Step 3.5.1. ----
  # Prepare [Train+{Movie*User}]_bias 
Train_MUb <- x %>% 
  left_join(movie_bias, by='movieId') %>%
  left_join(user_gb, by='userId')

  # Step 3.5.2. ----
  # calculate b_ug 
memory.limit(size = 7000)
b_ug <- Train_MUb 

for (i in 1:length(genres)){
  b_ug <- b_ug %>% 
    add_column(A[match(b_ug$movieId,gen$movieId),i]*
                 (b_ug$rating - mu - b_ug$b_i - b_ug$b_u))
}

colnames(b_ug)[6:(5+length(genres))] <- paste('R',genres, sep="_")
head(b_ug)

# Step 3.6. ----
# Calculate 'b_g', user bias for every genre, by user

b_g <- as.data.frame(list(userId = users$userId))

memory.limit(size = 7000)

for(index in 6:(5+length(genres))){    
  
  w <- as.data.frame(list(userId = b_ug$userId, b = b_ug[[index]]))
  names(w) <- c("userId","b")
    
  b_g <- b_g %>% 
    add_column( mean = w %>%
                  group_by(userId) %>%
                  summarize(UGmean=mean(b, na.rm=TRUE)) %>% .$UGmean)
}
 
names(b_g)[2:(1+length(genres))] <- genres 

head(b_g)



####################################################################
# STAGE 4: PREDICT RATINGS OVER THE TEST SET AND CALCULATE RMSE ####
####################################################################

# Step 4.1. ---- 
# MODEL1: Suming up all genre bias for each user

  # Step 4.1.1. ----
  # Predict Ratings
Test_MUb <- select(movieTest, movieId, userId) %>% 
  left_join(movie_bias, by='movieId') %>%
  left_join(user_gb, by='userId')  %>%
  mutate (alphaUser = match(userId,users$userId)) %>%
  mutate (alphaMovie = match(movieId,movies$movieId)) 

memory.limit(size = 8500)
D <- b_g[Test_MUb$alphaUser, 2:(1+length(genres))] * A[Test_MUb$alphaMovie,]

Test_MUb <- Test_MUb %>% cbind(b_g = rowSums(D, na.rm = TRUE))

predicted_ratings1 <- Test_MUb %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
      .$pred

  # Step 4.1.2. ----
  # RMSE
model_1_rmse <- RMSE(predicted_ratings1, movieTest$rating)
rmse_results <- data.frame(Method = "Genre Bias Sum", RMSE = model_1_rmse)

rmse_results

# write.csv(Test_MUb, "Test_Movie_1.csv", row.names=FALSE)

  # Step 4.1.3. ----
  # Histograms

    # Step 4.1.3.1 ----
    # Histogram of the movie bias distribution in the Test set; with Model 1
Test_MUb %>% 
  qplot(b_i, geom ="histogram",
        main = "Movie Bias Distribution | Test Set M1",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))

    # Step 4.1.3.2 ----
    # Histogram of the user bias distribution in the Test set; with Model 1
Test_MUb %>% 
  qplot(b_u, geom ="histogram",
        main = "User Bias Distribution | Test Set M1",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))

    # Step 4.1.3.3 ----
    # Histogram of the genre bias distribution in the Test set; with Model 1
Test_MUb %>% 
  qplot(b_g, geom ="histogram",
        main = "User-Genre Bias Distribution | Test Set M1",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))


# Step 4.2. ----
  # MODEL2: Mean of all genre bias for each user
  
  # Step 4.2.1 ----
  # Predict Ratings
Test_MUb2 <- select(movieTest, movieId, userId) %>% 
  left_join(movie_bias, by='movieId') %>%
  left_join(user_gb, by='userId')  %>%
  mutate (alphaUser = match(userId,users$userId)) %>%
  mutate (alphaMovie = match(movieId,movies$movieId)) 

memory.limit(size = 8500)
D <- b_g[Test_MUb2$alphaUser, 2:(1+length(genres))] * A[Test_MUb2$alphaMovie,]


Test_MUb2 <- Test_MUb2 %>% cbind(b_g = rowMeans(D, na.rm = TRUE))

Test_MUb2$b_g <- replace(Test_MUb2$b_g, which(is.na(Test_MUb2$b_g)),0)

predicted_ratings2 <- Test_MUb2 %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred


  # Step 4.2.2 ----
  # RMSE
model_2_rmse <- RMSE(predicted_ratings2, movieTest$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Genre Bias Mean",  
                                     RMSE = model_2_rmse ))
rmse_results

# write.csv(Test_MUb2, "Test_Movie_2.csv", row.names=FALSE)

  # Step 4.2.3. ----
  # Histograms
  
    # Step 4.2.3.1 ----
    # Histogram of the movie bias distribution in the Test set; with Model 2
Test_MUb2 %>% 
  qplot(b_i, geom ="histogram",
        main = "Movie Bias Distribution | Test Set M2",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))

    # Step 4.2.3.2 ----
    # Histogram of the user bias distribution in the Test set; with Model 2
Test_MUb2 %>% 
  qplot(b_u, geom ="histogram",
        main = "User Bias Distribution | Test Set M2",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))

    # Step 4.2.3.3 ----
    # Histogram of the genre bias distribution in the Test set; with Model 1
Test_MUb2 %>% 
  qplot(b_g, geom ="histogram",
        main = "User-Genre Bias Distribution | Test Set M2",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))



##############################################################
# STAGE 5: RUN MODEL OVER VALIDATION SET
##############################################################

# Step 5.1. ----
# Best of the latter models: MODEL2, Mean of all genre bias for each user
  
  # Step 5.2.1 ----
  # Predict Ratings

Val_MUb <- select(validation, movieId, userId) %>% 
  left_join(movie_bias, by='movieId') %>%
  left_join(user_gb, by='userId')  %>%
  mutate (alphaUser = match(userId,users$userId)) %>%
  mutate (alphaMovie = match(movieId,movies$movieId)) 

memory.limit(size = 8500)
V <- b_g[Val_MUb$alphaUser, 2:(1+length(genres))] * A[Val_MUb$alphaMovie,]


Val_MUb <- Val_MUb %>% cbind(b_g = rowMeans(V, na.rm = TRUE))

Val_MUb$b_g <- replace(Val_MUb$b_g, which(is.na(Val_MUb$b_g)),0)

predicted_ratingsVal <- Val_MUb %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

# write.csv(Val_MUb[,1:4], "Validation_Movie_PART_1.csv", row.names=FALSE)
# write.csv(Val_MUb[,5:7], "Validation_Movie_PART_2.csv", row.names=FALSE)

# Step 5.2. ----
# RMSE
Val_rmse <- RMSE(predicted_ratingsVal, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Genre Bias Mean (Validation)",  
                                     RMSE = Val_rmse ))

rmse_results %>% knitr::kable()

 write.csv(rmse_results, "rmse_results.csv", row.names=FALSE)

# Step 5.1.3. ----
# Histograms

# Step 5.1.3.1 ----
# Histogram of the movie bias distribution in the Validation set
Val_MUb %>% 
  qplot(b_i, geom ="histogram",
        main = "Movie Bias Distribution | Validation Set",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))

# Step 5.1.3.2 ----
# Histogram of the user bias distribution in the Validation set
Val_MUb %>% 
  qplot(b_u, geom ="histogram",
        main = "User Bias Distribution | Validation Set",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))

# Step 5.1.3.3 ----
# Histogram of the genre bias distribution in the Validation set
Val_MUb %>% 
  qplot(b_g, geom ="histogram",
        main = "User-Genre Bias Distribution | Validation Set",
        ylab = "Count",
        bins = 120, 
        data = ., 
        color = I("black"))
