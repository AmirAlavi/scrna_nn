library(limma)
library(edgeR)

# Load R objects containing data into workspace
load("counts_mat.gzip")		# counts_mat_r
load("accessions.gzip") 	# accessions_r
load("gene_symbols.gzip")	# gene_symbols_r
load("labels.gzip")		# labels_r
print("Done loading data objects")

counts_mat <- data.matrix(counts_mat_r, rownames.force=TRUE)
counts_mat <- t(counts_mat)
rownames(counts_mat) <- unname(gene_symbols_r)
print(str(counts_mat))
print(dim(counts_mat))

print("Filtering...")
# Remove cells with fewer than 1.8e3 lib size
csums <- colSums(counts_mat)
filter <- csums >= 1.8e3
cat("Number of filtered out cells: ", sum(!filter), "\n")
counts_mat <- counts_mat[,filter]
labels_r <- labels_r[filter]
accessions_r <- accessions_r[filter]
# Remove genes which have less than 10 reads accross all samples
rsums <- rowSums(counts_mat)
filter <- rsums >= 10
cat("Number of filtered out genes: ", sum(!filter), "\n")
counts_mat <- counts_mat[filter,]

type_factor_vector <- factor(labels_r)
print(table(type_factor_vector))
accessions_factor_vector <- factor(accessions_r)
print(table(accessions_factor_vector))



# Iterate over the types
# For each one, we need to create some lists:
#     others = all other cells that do not include the current type
#     grp1, .., grpN = lists of cells of current type, grouped by experiment.
#     	    	     Number of groups will depend on how many experiments had this cell type

for(type in levels(type_factor_vector)){
	 cat("Current type: ", type, "\n")
	 name_for_results_file <- strsplit(type, split=" ")[[1]][1]
	 cur_type_selection_vector <- type_factor_vector == type
	 other_types_selection_vector <- !cur_type_selection_vector
	 cat("\t", "Number of ", type, " cells: ", sum(cur_type_selection_vector), "\n")
	 cat("\t", "Number of remaining cells: ", sum(other_types_selection_vector), "\n")
	 
	 cur_type_experiments = accessions_factor_vector[cur_type_selection_vector]
	 uniq_cur_type_experiments = unique(cur_type_experiments)
	 cat("\t", "Number of different experiments for ", type, " ", length(uniq_cur_type_experiments), "\n")
	 exp_count <- 1
	 for(exp in uniq_cur_type_experiments){
	 	 name_for_results_file <- paste(name_for_results_file, exp_count, sep="_")
		 name_for_results_file <- paste(name_for_results_file, ".txt", sep="")
	 	 cur_exp_selection_vector <- accessions_factor_vector == exp

		 # Create two counts matrices
		 other_types_counts = counts_mat[, other_types_selection_vector]
		 cur_type_counts = counts_mat[, cur_type_selection_vector]

		 # Create a DGEList object for limma
		 combined_counts <- cbind(other_types_counts, cur_type_counts)
		 design <- cbind(OTHER=1, CURvsOTHER=rep(0:1, c(ncol(other_types_counts), ncol(cur_type_counts))))
		 #print("Creating DGEList object...")
		 #dge <- DGEList(combined_counts)

		 # Scale normalization using TMM method (suggested by limma docs)
		 # TODO: Docs say that it can be done without this normalization. Should we do it?
		 #print("Scale normalization using TMM...")
		 #dge <- calcNormFactors(dge)
		 
		 print("Voom...")
		 v <- voom(combined_counts, design, plot=FALSE)
		 print("lmFit...")
		 fit <-lmFit(v, design)
		 print("eBayes...")
		 fit <- eBayes(fit)
		 print("topTable...")
		 results <- topTable(fit, coef="CURvsOTHER", p.value=0.05, number=Inf)
		 print(typeof(results))
		 print(results)
		 write.table(results, name_for_results_file, sep="\t")
	 }
}

#library(DESeq2)