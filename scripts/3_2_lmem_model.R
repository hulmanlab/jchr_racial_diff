#clear all data frames from environment
rm(list=ls(all=TRUE)[sapply(mget(ls(all=TRUE)), class) == "data.frame"])

#clear all lists from environment
rm(list=ls(all=TRUE)[sapply(mget(ls(all=TRUE)), class) == "list"])

library(lme4)
library(Epi)

# load data
file_path_race = 'filepath/3_1_calculated_results_v6_race_lmem_v2.csv'
rmse_data <- read.csv(file_path_race)
# attach(rmse_data)
head(rmse_data) # show dataframe
# sum of column
column_base_mean <- mean(rmse_data$train_size_base, na.rm = TRUE)
column_base_median <- median(rmse_data$train_size_base, na.rm = TRUE)
column_single_tl_mean <- mean(rmse_data$train_size_single_tl, na.rm = TRUE)
column_single_tl_median <- median(rmse_data$train_size_single_tl, na.rm = TRUE)
df_length <- nrow(rmse_data)

print(paste("Mean_base: ", column_base_mean))
print(paste("Median_base: ", column_base_median))
print(paste("Mean_single_tl: ", column_single_tl_mean))
print(paste("Median_single_tl: ", column_single_tl_median))
print(paste("Number of rows: ", df_length))

rmse_data$train_size_naive <- 672 # 1 week with 15 min sampling frequency = 672


# adding column to rmse_data named race_w and race_b. containing B and W letters 
# - note that levels and labels are switched between the two
rmse_data$race_W <- factor(rmse_data$Race,  # factor(...): This function is used to encode a vector as a factor (categorical variable)
                           levels = c('white','black'),
                           labels = c('W','B'))

rmse_data$race_B <- factor(rmse_data$Race, 
                           levels = c('black','white'),
                           labels = c('b','w'))

rmse_data$age_C <- factor(rmse_data$AgeGroup, 
                          levels = c(1,2),
                          labels = c('c','a'))

rmse_data$age_A <- factor(rmse_data$AgeGroup, 
                          levels = c(2,1),
                          labels = c('a','c'))

rmse_data$sex_M <- factor(rmse_data$Gender, 
                          levels = c('M','F'),
                          labels = c('m','f'))

rmse_data$sex_F <- factor(rmse_data$Gender, 
                          levels = c('F','M'),
                          labels = c('f','m'))

rmse_data$ratio10 <- rmse_data$ratio/10

#############################################################
#################### continuous ratio #######################
#############################################################

# Adjusting for sample size
# explain training size: We are centering and rescaling
# we say that training size is about 510000 and because this value is much bigger then
# age, sex and the other variables then the model do not like it so we set everything to zero
# basically we are checking what the values are at a training size of 510000 or 4400


# # # GENERALIZED MODEL PART # # # comment in and comment out the fine-tuned part
# model <- "base"
# train_size_model <- "base"
# train_size_value <- 100000
# constant_value <- 5.1

# formula <- substitute(
#   rmse_var ~ ratio10*race_W +
#     age_A +
#     sex_M +
#     I(train_size_model / train_size_value - constant_value) +
#     (1 | PtID),
#   list(rmse_var = as.name(paste0("rmse_", model)),
#        train_size_model = as.name(paste0("train_size_", train_size_model)),
#        train_size_value = train_size_value,
#        constant_value = constant_value)
#   
# )

# ratio at 0 to 100 for all but naive
# male = 43% (100-43 = 57) (male = 0) => 0.5707 (to increase the representation of males to 50%)
# child = 40% 

# ratio at 0,50,100 for generalized model

# w_0  <- c(1,0, 0, 0.4, 0.5707,0,0)
# w_10 <- c(1,1, 0, 0.4, 0.5707,0,0)
# w_20 <- c(1,2, 0, 0.4, 0.5707,0,0)
# w_30 <- c(1,3, 0, 0.4, 0.5707,0,0)
# w_40 <- c(1,4, 0, 0.4, 0.5707,0,0)
# w_50 <- c(1,5, 0, 0.4, 0.5707,0,0)
# w_60 <- c(1,6, 0, 0.4, 0.5707,0,0)
# w_70 <- c(1,7, 0, 0.4, 0.5707,0,0)
# w_80 <- c(1,8, 0, 0.4, 0.5707,0,0)
# w_90 <- c(1,9, 0, 0.4, 0.5707,0,0)
# w_100<- c(1,10,0, 0.4, 0.5707,0,0)
# 
# 
# b_0  <- c(1,0, 1, 0.4, 0.5707, 0,0)
# b_10 <- c(1,1, 1, 0.4, 0.5707, 0,1)
# b_20 <- c(1,2, 1, 0.4, 0.5707, 0,2)
# b_30 <- c(1,3, 1, 0.4, 0.5707, 0,3)
# b_40 <- c(1,4, 1, 0.4, 0.5707, 0,4)
# b_50 <- c(1,5, 1, 0.4, 0.5707, 0,5)
# b_60 <- c(1,6, 1, 0.4, 0.5707, 0,6)
# b_70 <- c(1,7, 1, 0.4, 0.5707, 0,7)
# b_80 <- c(1,8, 1, 0.4, 0.5707, 0,8)
# b_90 <- c(1,9, 1, 0.4, 0.5707, 0,9)
# b_100<- c(1,10,1, 0.4, 0.5707, 0,10)




# # # FINE-TUNED MODEL PART # # # comment in and comment out the generalized model part

model <- "tl"
train_size_model <- "base"
train_size_value <- 100000
constant_value <- 5.1
train_size_model_tl <- "single_tl"
train_size_value_tl <- 1000
constant_value_tl <- 4.4

formula <- substitute(
  rmse_var ~ ratio10*race_W +
    age_A +
    sex_M +
    I(train_size_model / train_size_value - constant_value) +
    I(train_size_model_tl / train_size_value_tl - constant_value_tl) +
    (1 | PtID),
  list(rmse_var = as.name(paste0("rmse_", model)),
       train_size_model = as.name(paste0("train_size_", train_size_model)),
       train_size_value = train_size_value,
       constant_value = constant_value,
       
       train_size_model_tl = as.name(paste0("train_size_", train_size_model_tl)),
       train_size_value_tl = train_size_value_tl,
       constant_value_tl = constant_value_tl)
  
)


model_W_a_m <- lmer(formula, data = rmse_data)

ci.lin(model_W_a_m)
round(ci.lin(model_W_a_m),2)



# ratio at 0,50,100 for fine-tuned model
w_0  <- c(1,0, 0, 0.4, 0.5707,0,0,0)
w_10 <- c(1,1, 0, 0.4, 0.5707,0,0,0)
w_20 <- c(1,2, 0, 0.4, 0.5707,0,0,0)
w_30 <- c(1,3, 0, 0.4, 0.5707,0,0,0)
w_40 <- c(1,4, 0, 0.4, 0.5707,0,0,0)
w_50 <- c(1,5, 0, 0.4, 0.5707,0,0,0)
w_60 <- c(1,6, 0, 0.4, 0.5707,0,0,0)
w_70 <- c(1,7, 0, 0.4, 0.5707,0,0,0)
w_80 <- c(1,8, 0, 0.4, 0.5707,0,0,0)
w_90 <- c(1,9, 0, 0.4, 0.5707,0,0,0)
w_100<- c(1,10,0, 0.4, 0.5707,0,0,0)


b_0  <- c(1,0, 1, 0.4, 0.5707, 0,0,0) 
b_10 <- c(1,1, 1, 0.4, 0.5707, 0,0,1)
b_20 <- c(1,2, 1, 0.4, 0.5707, 0,0,2)
b_30 <- c(1,3, 1, 0.4, 0.5707, 0,0,3)
b_40 <- c(1,4, 1, 0.4, 0.5707, 0,0,4)
b_50 <- c(1,5, 1, 0.4, 0.5707, 0,0,5)
b_60 <- c(1,6, 1, 0.4, 0.5707, 0,0,6)
b_70 <- c(1,7, 1, 0.4, 0.5707, 0,0,7)
b_80 <- c(1,8, 1, 0.4, 0.5707, 0,0,8)
b_90 <- c(1,9, 1, 0.4, 0.5707, 0,0,9)
b_100<- c(1,10,1, 0.4, 0.5707, 0,0,10)

############# Show results from models#########################


# white
white_male = ci.lin(model_W_a_m, ctr.mat = rbind(
    w_0,w_10,w_20,w_30,w_40,w_50,w_60,w_70,w_80,w_90,w_100)
  )
round(
  ci.lin(model_W_a_m, ctr.mat = rbind(
    w_0,w_10,w_20,w_30,w_40,w_50,w_60,w_70,w_80,w_90,w_100)
  ), 2)

# black
black_male =
  ci.lin(model_W_a_m, ctr.mat = rbind(
    b_0,b_10,b_20,b_30,b_40,b_50,b_60,b_70,b_80,b_90,b_100)
  )
round(
  ci.lin(model_W_a_m, ctr.mat = rbind(
    b_0,b_10,b_20,b_30,b_40,b_50,b_60,b_70,b_80,b_90,b_100)
  ), 2)


# Define the contrast matrix names
contrast_names <- c(
  paste0("w_", seq(0, 100, by = 10)),
  paste0("b_", seq(0, 100, by = 10))
)

# Combine all the results into one data frame with contrast names
combined_results <- data.frame(
  Contrast = contrast_names,
  rbind(white_male, black_male)
)

# Optionally, save the results to a CSV file
# write.csv(combined_results, "filepath/filename", row.names = FALSE)

# ############################## Compare differences between white and black ##########################################

round(
  ci.lin(model_W_a_m, ctr.mat = rbind(
    w_0-b_0,
    w_50-b_50,
    w_100-b_100)
  ), 3)

# ################################ compare differences of the differences ##############################


round(
  ci.lin(model_W_a_m, ctr.mat = rbind(
    w_100 - w_0,
    b_100 - b_0,
    (w_100 - w_0)-(b_100 - b_0))
  ), 3)








