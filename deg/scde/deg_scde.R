library(scde)

n.cores <- 16
cat("Using ", n.cores, " cores", "\n")

# Load R objects containing data into workspace
load("counts_mat.gzip")		# counts_mat_r
load("accessions.gzip") 	# accessions_r
load("gene_symbols.gzip")	# gene_symbols_r
load("labels.gzip")		# labels_r
print("Done loading data objects")

counts_mat <- data.matrix(counts_mat_r, rownames.force=TRUE)
counts_mat <- t(counts_mat)
#rownames(counts_mat) <- unname(gene_symbols_r)
print(str(counts_mat))
print(dim(counts_mat))

print("Filtering...")
# Remove cells with fewer than 1.8e3 lib size
csums <- colSums(counts_mat > 0)
filter <- csums >= 1.8e3
cat("Number of filtered out cells, detected less than 1.8e3 genes: ", sum(!filter), "\n")
counts_mat <- counts_mat[,filter]
labels_r <- labels_r[filter]
accessions_r <- accessions_r[filter]
# Remove genes which have less than 10 reads accross all samples
rsums <- rowSums(counts_mat)
filter <- rsums >= 10
cat("Number of filtered out genes, have less than 10 reads across all samples: ", sum(!filter), "\n")
counts_mat <- counts_mat[filter,]
gene_symbols_r <- gene_symbols_r[filter]
# Remove genes that aren't detected in at least 5 cells
rsums <- rowSums(counts_mat > 0)
filter <- rsums >= 5
cat("Number of filtered out genes, not detected in at least 5 cells: ", sum(!filter), "\n")
counts_mat <- counts_mat[filter,]
gene_symbols_r <- gene_symbols_r[filter]

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
                 exp_count <- exp_count + 1

	 	 cur_exp_selection_vector <- accessions_factor_vector == exp

		 # Create two counts matrices
		 other_types_counts = counts_mat[, other_types_selection_vector]
		 cur_type_counts = counts_mat[, cur_type_selection_vector]

                 ## Sample at most 100 from each group
                 num_sample <- 100
                 if (ncol(cur_type_counts) > num_sample) {
		    cat("\t", "Many of current type, so sampling ", num_sample, "\n")
		    cur_type_counts <- cur_type_counts[, sample(ncol(cur_type_counts), num_sample)]
		 }
		 if (ncol(other_types_counts) > ncol(cur_type_counts)) {
		    cat("\t", "Many OTHER, so sampling ", ncol(cur_type_counts), "\n")
		    other_types_counts <- other_types_counts[, sample(ncol(other_types_counts), ncol(cur_type_counts))]
		 }

		 
 		 combined_counts <- cbind(other_types_counts, cur_type_counts)
                 print(str(combined_counts))

		 grouping <- rep(c("OTHERS", type), c(ncol(other_types_counts), ncol(cur_type_counts)))
		 groups <- factor(grouping)

		 cat("\t", "Fitting error models...", "\n")
		 t0 <- proc.time()
		 scde.fitted.model <- scde.error.models(counts=combined_counts, groups=groups, n.cores=n.cores, save.model.plots=F)
		 print(proc.time() - t0)
		 scde.prior <- scde.expression.prior(models=scde.fitted.model,counts=combined_counts)

                 cat("\t", "Calculating differential expression...", "\n")
                 t0 <- proc.time()
		 ediff <- scde.expression.difference(scde.fitted.model,combined_counts,scde.prior,groups=groups,n.cores=n.cores)
                 print(proc.time() - t0)
                 
		 p.values <- 2*pnorm(abs(ediff$Z),lower.tail=F) # 2-tailed p-value
		 p.values.adj <- 2*pnorm(abs(ediff$cZ),lower.tail=F) # Adjusted to control for FDR
		 significant.genes <- which(p.values.adj<0.05)
		 cat("\t", "num significant genes: ", length(significant.genes), "\n")
                 
                 ord <- order(p.values.adj[significant.genes]) # order by p-value
                 ## ord <- order(ediff[significant.genes, 2]) # order by log2 fold change
		 de <- cbind(names(gene_symbols_r[significant.genes]),gene_symbols_r[significant.genes],ediff[significant.genes,1:3],p.values.adj[significant.genes])[ord,]
		 colnames(de) <- c("EntrezID","symbol", "Lower bound","log2 fold change","Upper bound","p-value")
                 print(str(de))

		 print("Top 20 significant DEGs by p-value:")
		 print(de[1:20,])

		 # Separate up and down regulated:
		 upreg <- which(de[, 4]>0.0)
		 dwnreg <- which(de[, 4]<0.0)
		 
		 cat("\t", "num significant upreg genes: ", length(upreg), "\n")
		 ord <- order(de[upreg, 4])
		 ##up_de <- cbind(names(gene_symbols_r[upreg]),gene_symbols_r[upreg],ediff[upreg,1:3],p.values.adj[upreg])[ord,]
                 up_de <- de[upreg,][ord,]
 		 ##colnames(up_de) <- c("EntrezID","symbol","Lower bound","log2 fold change","Upper bound","p-value")
		 
 		 cat("\t", "num significant downreg genes: ", length(dwnreg), "\n")
		 ord <- order(de[dwnreg, 4])
		 ##dwn_de <- cbind(names(gene_symbols_r[dwnreg]),gene_symbols_r[dwnreg],ediff[dwnreg,1:3],p.values.adj[dwnreg])[ord,]
                 dwn_de <- de[dwnreg,][ord,]
 		 ##colnames(dwn_de) <- c("EntrezID","symbol","Lower bound","log2 fold change","Upper bound","p-value")
		 
		 name_for_results_file_all <- paste(name_for_results_file, "all", sep="_")
  		 name_for_results_file_all <- paste(name_for_results_file_all, ".csv", sep="")
		 name_for_results_file_up <- paste(name_for_results_file, "up", sep="_")
   		 name_for_results_file_up <- paste(name_for_results_file_up, ".csv", sep="")
		 name_for_results_file_dwn <- paste(name_for_results_file, "down", sep="_")
 		 name_for_results_file_dwn <- paste(name_for_results_file_dwn, ".csv", sep="")

		 write.table(de, name_for_results_file_all, sep=",", row.names=TRUE, col.names=TRUE)
 		 write.table(up_de, name_for_results_file_up, sep=",", row.names=TRUE, col.names=TRUE)
 		 write.table(dwn_de, name_for_results_file_dwn, sep=",", row.names=TRUE, col.names=TRUE)
	 }
}
