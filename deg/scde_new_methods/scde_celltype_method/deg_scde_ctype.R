args = commandArgs(trailingOnly=TRUE)
if (length(args) < 2) {
    stop("Missing args! arg1=data_dir, arg2=ncores", call.=FALSE)
} else {
    data_dir <- args[1]
    n.cores <- strtoi(args[2])
}

analyze_gene_counts <- function(counts_mat, labels_r) {
    print("In analyze_gene_counts...")
    type_factor_vector <- factor(labels_r)
    ## for each gene
    ##   for each cell type, calculate it's average red in that type. Cache it
    ## return a data structure which stores this average of averages for each gene.
    means_for_types <- list() # each element will be a dataframe that contains the means for a particular cell type
    insert_idx <- 1
    for(type in levels(type_factor_vector)){
        cat("Current type: ", type, "\n")

        cur_type_selection_vector <- type_factor_vector == type
        #print(str(cur_type_selection_vector))
        cur_type_counts_mat<- counts_mat[, cur_type_selection_vector]
        #print(str(cur_type_counts_mat))
        means <- rowMeans(cur_type_counts_mat)
        means_for_types[[insert_idx]] <- means
        insert_idx <- insert_idx + 1
        #other_types_selection_vector <- !cur_type_selection_vector
        #cat("\t", "Number of ", type, " cells: ", sum(cur_type_selection_vector), "\n")
        #cat("\t", "Number of remaining cells: ", sum(other_types_selection_vector), "\n" )
    }
    #print(str(means_for_types))
    combined_mat <- do.call(cbind, means_for_types)
    #print(str(combined_mat))
    final_means <- rowMeans(combined_mat)

    #print(str(final_means))
    png("average_of_avg_count_in_type.png")
    results <- hist(final_means, breaks=100, col="grey", main="Gene averages of (average count for type) for all types", xlab="Average of Averages")
    dev.off()
    print(results)
    ## png("average_of_avg_count_in_type_Sturges.png")
    ## hist(final_means, col="grey", main="Gene averages of (average count for type) for all types", xlab="Average of Averages")
    ## dev.off()

    ## find genes that have exactly 0. How many.
    thresh <- final_means == 0.0
    cat("Count of genes == 0.0: ", sum(thresh), "\n")
    ## find genes that have above 50, how many?
    thresh <- final_means > 50
    cat("Count of genes > 50: ", sum(thresh), "\n")
    ## find genes that have above 100, how many?
    thresh <- final_means > 100
    cat("Count of genes > 100: ", sum(thresh), "\n")
    ## find genes that have above 1,000, how many?
    thresh <- final_means > 1000
    cat("Count of genes > 1000: ", sum(thresh), "\n")
    ## find genes that have above 10,000, how many?
    thresh <- final_means > 10000
    cat("Count of genes > 10,000: ", sum(thresh), "\n")
    ## find genes that have above 20,000, how many?
    thresh <- final_means > 20000
    cat("Count of genes > 20,000: ", sum(thresh), "\n")
    ## find genes that have above 30,000, how many?
    thresh <- final_means > 30000
    cat("Count of genes > 30,000: ", sum(thresh), "\n")

    thresh <- final_means <= 100
    cat("Count of genes <= 100: ", sum(thresh), "\n")
    thresh_means <- final_means[thresh]
    cat("Histogram of genes <= 100:\n")
    png("average_of_avg_count_in_type_thresh_00100.png")
    results <- hist(thresh_means, breaks=100, col="grey", main="Gene averages of (average count/type) all types, capped @ 100", xlab="Average of Averages")
    dev.off()
    print(results)
    ## Try with a median of the averages
    cat("With medians of averages (instead of averages of averages):\n")
    final_medians <- apply(combined_mat, 1, median)
    png("median_of_avg_count_in_type.png")
    results <- hist(final_medians, breaks=100, col="grey", main="Gene medians of (average count for type) for all types", xlab="Median of Averages")
    dev.off()
    print(results)

    thresh <- final_medians <= 100
    cat("Count of genes (medians) <= 100: ", sum(thresh), "\n")
    thresh_medians <- final_medians[thresh]
    cat("Histogram of genes (medians)<= 100:\n")
    png("median_of_avg_count_in_type_thresh_00100.png")
    results <- hist(thresh_medians, breaks=100, col="grey", main="Gene medians of (average count/type) all types, capped @ 100", xlab="Median of Averages")
    dev.off()
    print(results)

    ## Take a cutoff on genes:
    thresh <- final_means > 50
    return(thresh)
}

plot_pval_hist <- function(pvals, filename) {
    png(filename)
    hist(pvals, breaks=100, col="grey", main=paste(filename, "DE raw p-vals", sep=" "), xlab="Raw p-values")
    dev.off()
}

library(scde)

cat("Using ", n.cores, " cores", "\n")
cat("Looking in ", data_dir, "\n") # Must have a data_dir variable in workspace!
## Load R objects containing data into workspace
load(paste(data_dir, "/", "counts_mat.gzip", sep=""))   # counts_mat_r
load(paste(data_dir, "/", "accessions.gzip", sep=""))   # accessions_r
load(paste(data_dir, "/", "gene_symbols.gzip", sep="")) # gene_symbols_r
load(paste(data_dir, "/", "labels.gzip", sep=""))       # labels_r
print("Done loading data objects")

counts_mat <- data.matrix(counts_mat_r, rownames.force=TRUE)
counts_mat <- t(counts_mat)
##rownames(counts_mat) <- unname(gene_symbols_r)
print(str(counts_mat))
print(dim(counts_mat))

print("Filtering...")
gene_thresh_selection_vector <- analyze_gene_counts(counts_mat, labels_r)
cat("Number of filtered out genes, have less than 50 of avg_avg_count_in_type reads: ", sum(!gene_thresh_selection_vector), "\n")
counts_mat <- counts_mat[gene_thresh_selection_vector,]
gene_symbols_r <- gene_symbols_r[gene_thresh_selection_vector]

## Remove cells with fewer than 1.8e3 lib size
csums <- colSums(counts_mat > 0)
filter <- csums >= 1.8e3
cat("Number of filtered out cells, detected less than 1.8e3 genes: ", sum(!filter), "\n")
counts_mat <- counts_mat[,filter]
labels_r <- labels_r[filter]
accessions_r <- accessions_r[filter]

## ## Remove genes which have less than 10 reads accross all samples
## rsums <- rowSums(counts_mat)
## filter <- rsums >= 10
## cat("Number of filtered out genes, have less than 10 reads across all samples: ", sum(!filter), "\n")
## counts_mat <- counts_mat[filter,]
## gene_symbols_r <- gene_symbols_r[filter]
## ## Remove genes that aren't detected in at least 5 cells
## rsums <- rowSums(counts_mat > 0)
## filter <- rsums >= 5
## cat("Number of filtered out genes, not detected in at least 5 cells: ", sum(!filter), "\n")
## counts_mat <- counts_mat[filter,]
## gene_symbols_r <- gene_symbols_r[filter]

type_factor_vector <- factor(labels_r)
print(table(type_factor_vector))
accessions_factor_vector <- factor(accessions_r)
print(table(accessions_factor_vector))

## Iterate over the types
## For each one, we need to create some lists:
##     others = all other cells that do not include the current type
##     grp1, .., grpN = lists of cells of current type, grouped by experiment.
##     	    	     Number of groups will depend on how many experiments had this cell type

for(type in levels(type_factor_vector)){

    cat("Current type: ", type, "\n")

    cur_type_selection_vector <- type_factor_vector == type
    other_types_selection_vector <- !cur_type_selection_vector
    cat("\t", "Number of ", type, " cells: ", sum(cur_type_selection_vector), "\n")
    cat("\t", "Number of remaining cells: ", sum(other_types_selection_vector), "\n")
    
    cur_type_experiments <- accessions_factor_vector[cur_type_selection_vector]
    uniq_cur_type_experiments <- unique(cur_type_experiments)
    cat("\t", "Number of different experiments for ", type, " ", length(uniq_cur_type_experiments), "\n")

    ## Collect the results of SCDE differential expression for each study, into a list for this cell type
    deg_results <- list()
    result_insert_idx <- 1
    for(exp in uniq_cur_type_experiments){
    
        cat("\t","\t", "Current experiment: ", exp, "\n")
        filename <- strsplit(type, split=" ")[[1]][1]
        filename <- paste(filename, exp, sep="_")

        ## Selection vector to extract samples for just this current experiment 
        cur_exp_selection_vector <- accessions_factor_vector == exp
        ## Selection vector to extract samples for just this current experiment AND this current cell type
        cur_type_exp_sv <- cur_exp_selection_vector & cur_type_selection_vector

        ## Ensure that each the sample size is large enough, if not, report that it wasn't and move on to next experiment
        min_sample_size <- 100
        num_samples <- sum(cur_type_exp_sv)
        cat("\t", "\t", "Num samples: ", num_samples, "\n")
        if (num_samples< min_sample_size) {
            cat("\t", "\t", "Too few samples, skipping experiment", "\n")
            next
        }

        ## Create two counts matrices
        other_types_counts = counts_mat[, other_types_selection_vector]
        cur_type_counts = counts_mat[, cur_type_exp_sv]

        ## Sample at most 100 from each group
        num_sample <- 100
        if (ncol(cur_type_counts) > num_sample) {
            cat("\t", "\t", "Many of current type, so sampling ", num_sample, "\n")
            cur_type_counts <- cur_type_counts[, sample(ncol(cur_type_counts), num_sample)]
        }
        if (ncol(other_types_counts) > ncol(cur_type_counts)) {
            cat("\t", "\t", "Many OTHER, so sampling ", ncol(cur_type_counts), "\n")
            #other_types_counts <- other_types_counts[, sample(ncol(other_types_counts), ncol(cur_type_counts))]
            
            # a different sampling method: 
            # ensure every "other type" got sampled; within each "other type", sample randomly
            num_types <- length(levels(type_factor_vector))
            if (num_types >= 100){
                # just use old sampling method
                other_types_counts <- other_types_counts[, sample(ncol(other_types_counts), ncol(cur_type_counts))]
            }
            else{
                raw_num_samples_per_type <- as.integer(100/(num_types - 1))
                rem <- 100 - raw_num_samples_per_type*(num_types - 1)
            
                other_types_counts <- c()
            
                for(o_type in levels(type_factor_vector)){
                    if (o_type != type){
                        if (rem > 0){
                            # remaindar correction for sample size per other type
                            num_samples_per_type <- raw_num_samples_per_type + 1
                            rem <- rem - 1
                        }
                        else{
                            num_samples_per_type <- raw_num_samples_per_type
                        }
                  
                        o_type_selection_vector <- type_factor_vector == o_type
                        o_type_counts = counts_mat[, o_type_selection_vector]
                        
                        num_o_type_sampled_cells <- min(ncol(o_type_counts), num_samples_per_type)
                        print(o_type)
                        cat("Number of other type sampled cells: ", num_o_type_sampled_cells, "\n")
                    
                        # randomly sample within current o_type
                        o_type_counts <- o_type_counts[, sample(ncol(o_type_counts), num_o_type_sampled_cells)]
                    
                        other_types_counts <- cbind(other_types_counts, o_type_counts)
                    
                    }
                }
                cat("Size of other types counts before final correction:", ncol(other_types_counts), "\n")
                
                # at the end, if total number of sampled cells < 100, randomly sample the remaining quota amoung all other types
                if (ncol(other_types_counts) < 100){
                    extra_cells = 100 - ncol(other_types_counts)
                  
                    all_other_types_counts = counts_mat[, other_types_selection_vector]

                    all_other_types_counts <- all_other_types_counts[, sample(ncol(all_other_types_counts), extra_cells)]
                  
                    other_types_counts <- cbind(other_types_counts, all_other_types_counts)
                }
            }
        }
        cat("Size of other types counts after final correction: ", ncol(other_types_counts), "\n")

        combined_counts <- cbind(other_types_counts, cur_type_counts)

        cat("\t", "\t", "str(combined_counts):", "\n")
        print(str(combined_counts))

        grouping <- rep(c("OTHERS", type), c(ncol(other_types_counts), ncol(cur_type_counts)))
        groups <- factor(grouping)

        cat("\t", "\t", "Fitting error models...", "\n")
        t0 <- proc.time()
        scde.fitted.model <- scde.error.models(counts=combined_counts, groups=groups, n.cores=n.cores, save.model.plots=F)
        print(proc.time() - t0)
        scde.prior <- scde.expression.prior(models=scde.fitted.model,counts=combined_counts)

        cat("\t", "\t", "Calculating differential expression...", "\n")
        t0 <- proc.time()
        ediff <- scde.expression.difference(scde.fitted.model,combined_counts,scde.prior,groups=groups,n.cores=n.cores)
        print(proc.time() - t0)
        
        p.values <- 2*pnorm(abs(ediff$Z),lower.tail=F) # 2-tailed p-value
        p.values.adj <- 2*pnorm(abs(ediff$cZ),lower.tail=F) # Adjusted to control for FDR

        ## TODO: plot histogram of the raw p-values from this experiment.
        plot_pval_hist(p.values, paste(filename, ".png", sep=""))
        ##significant.genes <- which(p.values.adj<0.05)
        ##cat("\t", "num significant genes: ", length(significant.genes), "\n")
        
        ##ord <- order(p.values.adj[significant.genes]) # order by p-value
        ##de <- cbind(names(gene_symbols_r[significant.genes]),gene_symbols_r[significant.genes],ediff[significant.genes,1:3],p.values[significant.genes], p.values.adj[significant.genes])[ord,]
        ##colnames(de) <- c("EntrezID","Symbol", "Lower_bound","Log2_fold_change","Upper_bound","Raw_p_value", "Adj_p_value")

        de <- cbind(names(gene_symbols_r),gene_symbols_r,ediff[,1:3],p.values, p.values.adj)
        colnames(de) <- c("EntrezID", "Symbol", "Lower_bound", "Log2_fold_change", "Upper_bound", "Raw_p_value", "Adj_p_value")
        
        deg_results[[result_insert_idx]] <- de # Add the results of this study to the collection for this cell type
        result_insert_idx <- result_insert_idx + 1
        ## Also save the results of this experiment in its own file, to have on record

        filename <- paste(filename, ".csv", sep="")
        write.table(de, filename, sep=",", row.names=TRUE, col.names=TRUE)
    }
    
    ## Now, for this cell type, if we had more than one study, do a meta-analysis by taking maximum adjusted p-value as the p-value
    ## and the average of the log2(foldchange)
    if (length(deg_results) == 0) {
        cat("\t", "No experiments done for this type", "\n")
    } else {
        max_adj_pvals = integer(dim(counts_mat)[1])
        avg_fold_change = integer(dim(counts_mat)[1])
        for (results in deg_results) {
            max_adj_pvals <- pmax(max_adj_pvals, results[, "Adj_p_value"])
            avg_fold_change <- avg_fold_change + results[, "Log2_fold_change"]
        }
        avg_fold_change <- avg_fold_change / length(deg_results)

        ## Now, do significance and FDR adjustment using these new p-values
        ## TODO: should we re-adjust p-vals?
        ##meta_adj_pvals <- p.adjust(max_adj_pvals, method="BH")
        meta_adj_pvals <- max_adj_pvals
        sig_genes_sv<- meta_adj_pvals < 0.05
        cat("\t", "Num sig DEGs after meta analysis: ", sum(sig_genes_sv), "\n")

        ## Check consistency of Log2FoldChange sign accross experiments
        fold_change_signs = integer(dim(counts_mat)[1])
        for (results in deg_results) {
            fold_change_signs <- fold_change_signs + sign(results[, "Log2_fold_change"])
        }
        fold_change_consistent_sv <- abs(fold_change_signs) == length(deg_results)
        sig_and_consistent_sv <- sig_genes_sv & fold_change_consistent_sv
        cat("\t", "Num of sig DEGs with consistent foldchange accross experiments: ", sum(sig_and_consistent_sv), "\n")
        
        ord <- order(meta_adj_pvals[sig_and_consistent_sv]) # order by p-value
        final_results <- cbind(names(gene_symbols_r[sig_and_consistent_sv]), gene_symbols_r[sig_and_consistent_sv], avg_fold_change[sig_and_consistent_sv], meta_adj_pvals[sig_and_consistent_sv])[ord,]
        colnames(final_results) <- c("EntrezID", "Symbol", "Avg_log2_fold_change", "Max_adj_p_value")
        ## Finally, Save the results
        filename <- strsplit(type, split=" ")[[1]][1]
        filename <- paste(filename, "meta", sep="_")
        filename <- paste(filename, ".csv", sep="")
        write.table(final_results, filename, sep=",", row.names=TRUE, col.names=TRUE)

    }

}
