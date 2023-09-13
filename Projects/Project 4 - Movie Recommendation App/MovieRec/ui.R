## ui.R
library(shiny)
library(shinydashboard)
library(recommenderlab)
library(data.table)
library(ShinyRatingInput)
library(shinyjs)
library(markdown)

source('functions/helpers.R')

shinyUI(
    navbarPage("Movie Recommendation System",
        tabPanel("Recommendation Based on Ratings", 
            dashboardPage(
                skin = "blue",
                dashboardHeader(title = "Movie Recommender based on User Ratings"),
                
                dashboardSidebar(disable = TRUE),

                dashboardBody(includeCSS("css/movie.css"),
                    fluidRow(
                        box(width = 12, 
                            title = "Step 1: Rate as many movies as possible", 
                            status = "info", solidHeader = TRUE, collapsible = TRUE,
                            div(class = "rateitems",
                                uiOutput('ratings'))
                        )
                    ),
                    fluidRow(
                        useShinyjs(),
                        box(width = 12, status = "info", solidHeader = TRUE,
                            title = "Step 2: Discover movies you might like",
                            br(),
                            withBusyIndicatorUI(
                            actionButton("btn", "Click here to get your recommendations", class = "btn-warning")
                            ),
                            br(),
                            tableOutput("results")
                        )
                    )
                )
            )
        ),

        tabPanel("Recommendation Based on Genre",
            dashboardPage(
                skin = "blue",
                dashboardHeader(title = "Movie Recommender based on Genres"),
                dashboardSidebar(disable = TRUE),
                dashboardBody(includeCSS("css/movie.css"),
                    fluidRow(
                        box(width = 12, title = "Step 1: Select your favourate movie genre", 
                            status = "info", solidHeader = TRUE, collapsible = TRUE,
                            inputPanel(selectInput("Genres",
                                       label = "Select Your Movie Genre",
                                       choices = c("Action", "Adventure", "Animation", 
                                                   "Children's", "Comedy", "Crime",
                                                   "Documentary", "Drama", "Fantasy",
                                                   "Film-Noir", "Horror", "Musical", 
                                                   "Mystery", "Romance", "Sci-Fi", 
                                                   "Thriller", "War", "Western")))
                        )
                    ),
                    fluidRow(
                        useShinyjs(),
                        box(width = 12, status = "info", solidHeader = TRUE,
                            title = "Step 2: Discover movies you might like",
                            br(),
                            # withBusyIndicatorUI(
                            # actionButton("btn", "Click here to get your recommendations", class = "btn-warning")
                            # ),
                            br(),
                            tableOutput("results0")
                        )
                    )
                )
            )
        )
    )
) 