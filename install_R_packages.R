# List of required R packages
required_packages <- c(
  "devtools",
  "igraph",
  "mclust",
  "kernlab",
  "R.matlab",
  "bc3net",
  "optparse",
  "tidyverse"
)

# Function to install packages if not already installed
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Install CRAN packages
cran_packages <- setdiff(required_packages, "bc3net")
lapply(cran_packages, install_if_missing)

# Install bc3net from GitHub
if (!require("bc3net", character.only = TRUE)) {
  devtools::install_github("bc3net/bc3net")
}
