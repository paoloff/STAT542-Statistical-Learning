## server.R

# # load functions
library(recommenderlab)
library(ShinyRatingInput)

myurl = "https://liangfgithub.github.io/MovieData/"
ratings = read.csv(paste0(myurl, 'ratings.dat?raw=true'), 
                   sep = ':',
                   colClasses = c('integer', 'NULL'), 
                   header = FALSE)

colnames(ratings) = c('UserID', 'MovieID', 'Rating', 'Timestamp')
i = paste0('u', ratings$UserID)
j = paste0('m', ratings$MovieID)
x = ratings$Rating
tmp = data.frame(i, j, x, stringsAsFactors = T)
Rmat = sparseMatrix(as.integer(tmp$i), as.integer(tmp$j), x = tmp$x)
ncols = ncol(Rmat)
rownames(Rmat) = levels(tmp$i)
colnames(Rmat) = levels(tmp$j)
Rmat = new('realRatingMatrix', data = Rmat)

# define functions
get_user_ratings = function(value_list) {
  dat = data.table(MovieID = sapply(strsplit(names(value_list), "_"), function(x) ifelse(length(x) > 1, x[[2]], NA)),
                   Rating = unlist(as.character(value_list)))
  dat = dat[!is.null(Rating) & !is.na(MovieID)]
  dat[Rating == " ", Rating := 0]
  dat[, ':=' (MovieID = as.numeric(MovieID), Rating = as.numeric(Rating))]
  dat = dat[Rating > 0]
  
  mids <- dat[[1]]
  ratings <- dat[[2]]
  newuser = rep(NA, ncol(Rmat))
  movieIDs <- colnames(Rmat)

  for (i in 1:length(mids)) {
    newuser[which(movieIDs == paste0('m', mids[i]))] <- ratings[i]
  }

  test = matrix(newuser, 
                nrow=1, 
                ncol=ncol(Rmat),
                dimnames = list(user=paste('New'),
                                item=movieIDs)
                )
  return(test)
}


recom_by_genre_popularity = list("Action" = c(260, 1196, 1210, 480, 2028, 589, 2571, 1580, 1198, 110),
                                 "Adventure" = c(260, 1196, 1210, 480, 1580, 1198, 1197, 2628, 2916, 2987),
                                 "Animation" = c(1, 2987, 2355, 3114, 588, 3751, 2700, 364, 595, 2081),
                                 "Children's" = c(1097, 1, 34, 919, 2355, 3114, 588, 3751, 1073, 364),
                                 "Comedy" = c(2858, 1270, 1580, 2396, 1197, 1265, 2997, 356, 2716, 1),
                                 "Crime" = c(608, 1617, 858, 296, 50, 1221, 1213, 2000, 592, 1089),
                                 "Documentary" = c(2064, 246, 162, 3007, 1147, 1189, 2859, 2677, 2693, 1191),
                                 "Drama" = c(2858, 1196, 2028, 593, 608, 110, 527, 1097, 318, 858),
                                 "Fantasy" = c(260, 1097, 2628, 2174, 2797, 1073, 367, 2100, 2054, 2968),
                                 "Film-Noir" = c(1617, 541, 2987, 1252, 913, 1179, 1748, 1267, 3683, 3435),
                                 "Horror" = c(2716, 1214, 1387, 1219, 2710, 2657, 1278, 1258, 2455, 3081),
                                 "Musical" = c(919, 588, 1220, 2657, 364, 1288, 595, 2081, 1028, 551),
                                 "Mystery" = c(1617, 924, 648, 3176, 1252, 1732, 904, 913, 1909, 903),
                                 "Romance" = c(1210, 2396, 1197, 1265, 356, 912, 377, 1307, 1721, 2291),
                                 "Sci-Fi" = c(260, 1196, 1210, 480, 589, 2571, 1270, 1580, 1097, 2628),
                                 "Thriller" = c(589, 2571, 593, 608, 2762, 1617, 1240, 1214, 2916, 457),
                                 "War" = c(1196, 1210, 2028, 110, 527, 356, 1200, 780, 912, 750),
                                 "Western" = c(590, 1304, 2012, 3671, 1266, 2701, 1201, 368, 266, 2527))

pred_by_genre <- function(genre) {
  return(recom_by_genre_popularity[[genre]])
}

myurl = "https://liangfgithub.github.io/MovieData/"
movies = readLines(paste0(myurl, 'movies.dat?raw=true'))
movies = strsplit(movies, split = "::", fixed = TRUE, useBytes = TRUE)
movies = matrix(unlist(movies), ncol = 3, byrow = TRUE)
movies = data.frame(movies, stringsAsFactors = FALSE)
colnames(movies) = c('MovieID', 'Title', 'Genres')
movies$MovieID = as.integer(movies$MovieID)
movies$Title = iconv(movies$Title, "latin1", "UTF-8")

small_image_url = "https://liangfgithub.github.io/MovieImages/"
movies$image_url = sapply(movies$MovieID, 
                          function(x) paste0(small_image_url, x, '.jpg?raw=true'))

shinyServer(function(input, output, session) {
  
  # show the books to be rated
  output$ratings <- renderUI({
    num_rows <- 20
    num_movies <- 6 # movies per row
    
    lapply(1:num_rows, function(i) {
      list(fluidRow(lapply(1:num_movies, function(j) {
        list(box(width = 2,
                 div(style = "text-align:center", img(src = movies$image_url[(i - 1) * num_movies + j], height = 150)),
                 #div(style = "text-align:center; color: #999999; font-size: 80%", books$authors[(i - 1) * num_books + j]),
                 div(style = "text-align:center", strong(movies$Title[(i - 1) * num_movies + j])),
                 div(style = "text-align:center; font-size: 150%; color: #f0ad4e;", ratingInput(paste0("select_", movies$MovieID[(i - 1) * num_movies + j]), label = "", dataStop = 5)))) #00c0ef
      })))
    })
  })
  

  # Calculate recommendations when the sbumbutton is clicked
  df <- eventReactive(input$btn, {
    withBusyIndicatorServer("btn", { # showing the busy indicator
        # hide the rating container
        useShinyjs()
        jsCode <- "document.querySelector('[data-widget=collapse]').click();"
        runjs(jsCode)
        
        # get the user's rating data
        value_list <- reactiveValuesToList(input)
        user_ratings <- get_user_ratings(value_list)
        test <- as(user_ratings, "realRatingMatrix")

        # recommender.IBCF <- Recommender(Rmat, method = "IBCF",
        #                                 parameter = list(normalize = 'center', 
        #                                 method = 'Cosine', 
        #                                 k = 30))
        # saveRDS(recommender.IBCF, file = "rec.rds")

        recommender.IBCF <- readRDS("rec.rds")

        p.IBCF <- predict(recommender.IBCF, test, type="ratings")
        p.IBCF <- as.numeric(as(p.IBCF, "matrix"))

        recom_result <- tail(order(p.IBCF, decreasing = FALSE, na.last = FALSE), 10)
        recom_result <- rev(recom_result)

        user_predicted_ids <- as.numeric(gsub("m", "", colnames(Rmat)[recom_result]))

        user_predicted_ids = pmatch(user_predicted_ids, movies$MovieID)
        recom_results <- data.table(Rank = 1:10, 
                                    MovieID = movies$MovieID[user_predicted_ids], 
                                    Title = movies$Title[user_predicted_ids])
        
    }) # still busy
    
  }) # clicked on button


  df2 <- eventReactive(input$Genres, {
    user_predicted_ids <- pred_by_genre(input$Genres)
    user_predicted_ids <- pmatch(user_predicted_ids, movies$MovieID)

    genre_result <- data.table(Rank = 1:5, 
                                MovieID = user_predicted_ids, 
                                Title = movies$Title[user_predicted_ids])
    
  }) # clicked on button
  

  # display the recommendations
  output$results <- renderUI({
    num_rows <- 2
    num_movies <- 5
    recom_result <- df()
    
    lapply(1:num_rows, function(i) {
      list(fluidRow(lapply(1:num_movies, function(j) {
        box(width = 2, status = "success", solidHeader = TRUE, title = paste0("Rank ", (i - 1) * num_movies + j),
            
          div(style = "text-align:center", 
              a(img(src = movies$image_url[recom_result$MovieID[(i - 1) * num_movies + j]], height = 150))
             ),
          div(style="text-align:center; font-size: 100%", 
              strong(movies$Title[recom_result$MovieID[(i - 1) * num_movies + j]])
             )
          
        )        
      }))) # columns
    }) # rows
    
  }) # renderUI function

  output$results0 <- renderUI({
    num_rows <- 2
    num_movies <- 5
    genre_result <- df2()
    lapply(1:num_rows, function(i) {
      list(fluidRow(lapply(1:num_movies, function(j) {
        box(width = 2, status = "success", solidHeader = TRUE, title = paste0("Rank ", (i - 1) * num_movies + j),
            
            div(style = "text-align:center", 
                a(img(src = movies$image_url[genre_result$MovieID[(i - 1) * num_movies + j]], height = 150))
            ),
            div(style="text-align:center; font-size: 100%", 
                strong(movies$Title[genre_result$MovieID[(i - 1) * num_movies + j]])
            )
            
        )        
      }))) # columns
    }) # rows
    
  }) # renderUI function
  
}) # server function