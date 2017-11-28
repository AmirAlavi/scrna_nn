args = commandArgs(trailingOnly=TRUE)
if (length(args) < 3) {
  stop("Missing args! arg1=original_method_dir, arg2=distance_method_dir, arg3=celltype", call.=FALSE)
} else {
  original_method_dir <- args[1]
  distance_method_dir <- args[2]
  celltype <- args[3]
}

in_filename <- paste0(celltype, "_meta", ".csv")
out_filename <- paste0(celltype, "_compare", ".txt")

original_file <- paste0(original_method_dir, "/", in_filename)
distance_file <- paste0(distance_method_dir, "/", in_filename)

original_method <- read.delim(original_file, header=TRUE, sep=",")
distance_method <- read.delim(distance_file, header=TRUE, sep=",")

original_top100 <- original_method[1:100 ,]
distance_top100 <- distance_method[1:100 ,]

common_genes <- original_top100$Symbol[ original_top100$Symbol %in% distance_top100$Symbol]
unique_genes_original <- original_top100$Symbol[! original_top100$Symbol %in% distance_top100$Symbol]
unique_genes_distance <- distance_top100$Symbol[! distance_top100$Symbol %in% original_top100$Symbol]

file <- file(out_filename)
writeLines("----------------\nOverlapped Genes\n----------------", file)
close(file)
file <- file(out_filename, "a")
writeLines(as.character(common_genes), file)
close(file)
file <- file(out_filename, "a")
writeLines("\n-------------------------------\nUnique Genes in Distance Method\n-------------------------------", file)
close(file)
file <- file(out_filename, "a")
writeLines(as.character(unique_genes_distance), file)
close(file)
file <- file(out_filename, "a")
writeLines("\n-------------------------------\nUnique Genes in Original Method\n-------------------------------", file)
close(file)
file <- file(out_filename, "a")
writeLines(as.character(unique_genes_original), file)
close(file)


