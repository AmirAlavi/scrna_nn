library(scde)

n.cores <- 16

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
# Remove cells with fewer than 1.8e5 lib size
csums <- colSums(counts_mat)
filter <- csums >= 1.8e5
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

	 cur_type_selection_vector <- type_factor_vector == type
	 other_types_selection_vector <- !cur_type_selection_vector
	 cat("\t", "Number of ", type, " cells: ", sum(cur_type_selection_vector), "\n")
	 cat("\t", "Number of remaining cells: ", sum(other_types_selection_vector), "\n")
	 
	 cur_type_experiments = accessions_factor_vector[cur_type_selection_vector]
	 uniq_cur_type_experiments = unique(cur_type_experiments)
	 cat("\t", "Number of different experiments for ", type, " ", length(uniq_cur_type_experiments), "\n")
	 exp_count <- 1
	 for(exp in uniq_cur_type_experiments){
	 	 name_for_results_file <- strsplit(type, split=" ")[[1]][1]
	 	 name_for_results_file <- paste(name_for_results_file, exp_count, sep="_")

	 	 cur_exp_selection_vector <- accessions_factor_vector == exp

		 # Create two counts matrices
		 other_types_counts = counts_mat[, other_types_selection_vector]
		 cur_type_counts = counts_mat[, cur_type_selection_vector]
 		 combined_counts <- cbind(other_types_counts, cur_type_counts)

		 grouping <- rep(c("OTHERS", type), c(ncol(other_types_counts), ncol(cur_type_counts)))
		 groups <- factor(grouping)

		 cat("\t", "Fitting error models...", "\n")
		 t0 <- proc.time()
		 scde.fitted.model <- scde.error.models(counts=combined_counts, groups=groups, n.cores=n.cores, save.model.plots=F)
		 proc.time() - t0
		 scde.prior <- scde.expression.prior(models=scde.fitted.model,counts=combined_counts)

		 ediff <- scde.expression.difference(scde.fitted.model,counts,scde.prior,groups=groups,n.cores=n.cores)
		 p.values <- 2*pnorm(abs(ediff$Z),lower.tail=F) # 2-tailed p-value
		 p.values.adj <- 2*pnorm(abs(ediff$cZ),lower.tail=F) # Adjusted to control for FDR
		 significant.genes <- which(p.values.adj<0.05)
		 cat("\t", "num significant genes: ", length(significant.genes), "\n")

		 ord <- order(p.values.adj[significant.genes]) # order by p-value
		 de <- cbind(ediff[significant.genes,1:3],p.values.adj[significant.genes])[ord,]
		 colnames(de) <- c("Lower bound","log2 fold change","Upper bound","p-value")

		 print("Top 20 most significant DEGs:")
		 de[1:20,]

		 # Separate up and down regulated:
		 upreg <- which(ediff[significant.genes, 2]>0.0)
		 dwnreg <- which(ediff[significant.genes, 2]<0.0)
		 
		 cat("\t", "num significant upreg genes: ", length(upreg), "\n")
		 ord <- order(p.values.adj[upreg])
		 up_de <- cbind(ediff[upreg,1:3],p.values.adj[upreg])[ord,]
 		 colnames(up_de) <- c("Lower bound","log2 fold change","Upper bound","p-value")
		 
 		 cat("\t", "num significant downreg genes: ", length(dwnreg), "\n")
		 ord <- order(p.values.adj[dwnreg])
		 dwn_de <- cbind(ediff[dwnreg,1:3],p.values.adj[dwnreg])[ord,]
 		 colnames(dwn_de) <- c("Lower bound","log2 fold change","Upper bound","p-value")
		 
		 name_for_results_file_all <- paste(name_for_results_file, "all", sep="_")
  		 name_for_results_file_all <- paste(name_for_results_file_all, ".txt", sep="")
		 name_for_results_file_up <- paste(name_for_results_file, "up", sep="_")
   		 name_for_results_file_up <- paste(name_for_results_file_up, ".txt", sep="")
		 name_for_results_file_dwn <- paste(name_for_results_file, "down", sep="_")
 		 name_for_results_file_dwn <- paste(name_for_results_file_dwn, ".txt", sep="")

		 write.table(de, name_for_results_file_all, sep="\t")
 		 write.table(up_de, name_for_results_file_all_up, sep="\t")
 		 write.table(dwn_de, name_for_results_file_all_dwn, sep="\t")
	 }
}
