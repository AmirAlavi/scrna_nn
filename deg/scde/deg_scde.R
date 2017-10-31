## Poorman's command line argument parsing:
args = commandArgs(trailingOnly=TRUE)
if (length(args) < 3) {
    stop("Missing args! arg1=data.dir, arg2=ncores, arg3=max.coverage.bool", call.=FALSE)
} else {
    data.dir <- args[1]
    n.cores <- strtoi(args[2])
    do.max.coverage
}

analyze.gene.counts <- function(counts.mat, labels) {
    print("In analyze.gene.counts...")
    type.factor.vector <- factor(labels)
    ## for each gene
    ##   for each cell type, calculate it's average red in that type. Cache it
    ## return a data structure which stores this average of averages for each gene.
    means.for.types <- list() # each element will be a dataframe that contains the means for a particular cell type
    insert.idx <- 1
    for(type in levels(type.factor.vector)){
        cat("Current type: ", type, "\n")

        cur.type.selection.vector <- type.factor.vector == type
        #print(str(cur.type.selection.vector))
        cur.type.counts.mat<- counts.mat[, cur.type.selection.vector]
        #print(str(cur.type.counts.mat))
        means <- rowMeans(cur.type.counts.mat)
        means.for.types[[insert.idx]] <- means
        insert.idx <- insert.idx + 1
        #other.types.selection.vector <- !cur.type.selection.vector
        #cat("\t", "Number of ", type, " cells: ", sum(cur.type.selection.vector), "\n")
        #cat("\t", "Number of remaining cells: ", sum(other.types.selection.vector), "\n" )
    }
    #print(str(means.for.types))
    combined.mat <- do.call(cbind, means.for.types)
    #print(str(combined.mat))
    final.means <- rowMeans(combined.mat)

    #print(str(final.means))
    png("average_of_avg_count_in_type.png")
    results <- hist(final.means, breaks=100, col="grey", main="Gene averages of (average count for type) for all types", xlab="Average of Averages")
    dev.off()
    print(results)
    ## png("average_of_avg_count_in_type_Sturges.png")
    ## hist(final.means, col="grey", main="Gene averages of (average count for type) for all types", xlab="Average of Averages")
    ## dev.off()

    ## find genes that have exactly 0. How many.
    thresh <- final.means == 0.0
    cat("Count of genes == 0.0: ", sum(thresh), "\n")
    ## find genes that have above 50, how many?
    thresh <- final.means > 50
    cat("Count of genes > 50: ", sum(thresh), "\n")
    ## find genes that have above 100, how many?
    thresh <- final.means > 100
    cat("Count of genes > 100: ", sum(thresh), "\n")
    ## find genes that have above 1,000, how many?
    thresh <- final.means > 1000
    cat("Count of genes > 1000: ", sum(thresh), "\n")
    ## find genes that have above 10,000, how many?
    thresh <- final.means > 10000
    cat("Count of genes > 10,000: ", sum(thresh), "\n")
    ## find genes that have above 20,000, how many?
    thresh <- final.means > 20000
    cat("Count of genes > 20,000: ", sum(thresh), "\n")
    ## find genes that have above 30,000, how many?
    thresh <- final.means > 30000
    cat("Count of genes > 30,000: ", sum(thresh), "\n")

    thresh <- final.means <= 100
    cat("Count of genes <= 100: ", sum(thresh), "\n")
    thresh.means <- final.means[thresh]
    cat("Histogram of genes <= 100:\n")
    png("average_of_avg_count_in_type_thresh_00100.png")
    results <- hist(thresh.means, breaks=100, col="grey", main="Gene averages of (average count/type) all types, capped @ 100", xlab="Average of Averages")
    dev.off()
    print(results)
    ## Try with a median of the averages
    cat("With medians of averages (instead of averages of averages):\n")
    final.medians <- apply(combined.mat, 1, median)
    png("median_of_avg_count_in_type.png")
    results <- hist(final.medians, breaks=100, col="grey", main="Gene medians of (average count for type) for all types", xlab="Median of Averages")
    dev.off()
    print(results)

    thresh <- final.medians <= 100
    cat("Count of genes (medians) <= 100: ", sum(thresh), "\n")
    thresh_medians <- final.medians[thresh]
    cat("Histogram of genes (medians)<= 100:\n")
    png("median_of_avg_count_in_type_thresh_00100.png")
    results <- hist(thresh_medians, breaks=100, col="grey", main="Gene medians of (average count/type) all types, capped @ 100", xlab="Median of Averages")
    dev.off()
    print(results)

    ## Take a cutoff on genes:
    thresh <- final.means > 50
    return(thresh)
}

PlotPvalHist <- function(pvals, filename) {
    png(filename)
    hist(pvals, breaks=100, col="grey", main=paste(filename, "DE raw p-vals", sep=" "), xlab="Raw p-values")
    dev.off()
}

library(scde)

cat("Using ", n.cores, " cores", "\n")
cat("Looking in ", data.dir, "\n") # Must have a data.dir variable in workspace!
## Load R objects containing data into workspace
load(paste(data.dir, "/", "counts_mat.gzip", sep=""))   # counts_mat_r
load(paste(data.dir, "/", "accessions.gzip", sep=""))   # accessions_r
load(paste(data.dir, "/", "gene_symbols.gzip", sep="")) # gene_symbols_r
load(paste(data.dir, "/", "labels.gzip", sep=""))       # labels_r
print("Done loading data objects")

## Rename to comply with R coding style guides
counts.mat <- counts_mat_r
accessions <- accessions_r
gene.symbols <- gene_symbols_r
labels <- labels_r

counts.mat <- data.matrix(counts.mat, rownames.force=TRUE)
counts.mat <- t(counts.mat)
##rownames(counts.mat) <- unname(gene.symbols)
print(str(counts.mat))
print(dim(counts.mat))

print("Filtering...")
gene.thresh.selection.vector <- analyze.gene.counts(counts.mat, labels)
cat("Number of filtered out genes, have less than 50 of avg_avg_count_in_type reads: ", sum(!gene.thresh.selection.vector), "\n")
counts.mat <- counts.mat[gene.thresh.selection.vector,]
gene.symbols <- gene.symbols[gene.thresh.selection.vector]

## Remove cells with fewer than 1.8e3 lib size
csums <- colSums(counts.mat > 0)
filter <- csums >= 1.8e3
cat("Number of filtered out cells, detected less than 1.8e3 genes: ", sum(!filter), "\n")
counts.mat <- counts.mat[,filter]
labels <- labels[filter]
accessions <- accessions[filter]

type.factor.vector <- factor(labels)
print(table(type.factor.vector))
accessions.factor.vector <- factor(accessions)
print(table(accessions.factor.vector))

## Iterate over the types
## For each one, we need to create some lists:
##     others = all other cells that do not include the current type
##     grp1, .., grpN = lists of cells of current type, grouped by experiment.
##     	    	     Number of groups will depend on how many experiments had this cell type

for(type in levels(type.factor.vector)){
    cat("Current type: ", type, "\n")

    cur.type.selection.vector <- type.factor.vector == type
    other.types.selection.vector <- !cur.type.selection.vector
    cat("\t", "Number of ", type, " cells: ", sum(cur.type.selection.vector), "\n")
    cat("\t", "Number of remaining cells: ", sum(other.types.selection.vector), "\n")
    
    cur.type.experiments <- accessions.factor.vector[cur.type.selection.vector]
    uniq.cur.type.experiments <- unique(cur.type.experiments)
    cat("\t", "Number of different experiments for ", type, " ", length(uniq.cur.type.experiments), "\n")

    ## Collect the results of SCDE differential expression for each study, into a list for this cell type
    deg.results <- list()
    result.insert.idx <- 1
    for(exp in uniq.cur.type.experiments){
        cat("\t","\t", "Current experiment: ", exp, "\n")
        filename <- strsplit(type, split=" ")[[1]][1]
        filename <- paste(filename, exp, sep="_")

        ## Selection vector to extract samples for just this current experiment 
        cur.exp.selection.vector <- accessions.factor.vector == exp
        ## Selection vector to extract samples for just this current experiment AND this current cell type
        cur.type.exp.sv <- cur.exp.selection.vector & cur.type.selection.vector

        ## Ensure that each the sample size is large enough, if not, report that it wasn't and move on to next experiment
        min.sample.size <- 100
        num.samples <- sum(cur.type.exp.sv)
        cat("\t", "\t", "Num samples: ", num.samples, "\n")
        if (num.samples< min.sample.size) {
            cat("\t", "\t", "Too few samples, skipping experiment", "\n")
            next
        }

        ## Create two counts matrices
        other.types.counts = counts.mat[, other.types.selection.vector]
        cur.type.counts = counts.mat[, cur.type.exp.sv]

        ## Sample at most 100 from each group
        num.sample <- 100
        if (ncol(cur.type.counts) > num.sample) {
            cat("\t", "\t", "Many of current type, so sampling ", num.sample, "\n")
            cur.type.counts <- cur.type.counts[, sample(ncol(cur.type.counts), num.sample)]
        }
        if (ncol(other.types.counts) > ncol(cur.type.counts)) {
            cat("\t", "\t", "Many OTHER, so sampling ", ncol(cur.type.counts), "\n")
            other.types.counts <- other.types.counts[, sample(ncol(other.types.counts), ncol(cur.type.counts))]
        }

        combined.counts <- cbind(other.types.counts, cur.type.counts)
        cat("\t", "\t", "str(combined.counts):", "\n")
        print(str(combined.counts))

        grouping <- rep(c("OTHERS", type), c(ncol(other.types.counts), ncol(cur.type.counts)))
        groups <- factor(grouping)

        cat("\t", "\t", "Fitting error models...", "\n")
        t0 <- proc.time()
        scde.fitted.model <- scde.error.models(counts=combined.counts, groups=groups, n.cores=n.cores, save.model.plots=F)
        print(proc.time() - t0)
        scde.prior <- scde.expression.prior(models=scde.fitted.model,counts=combined.counts)

        cat("\t", "\t", "Calculating differential expression...", "\n")
        t0 <- proc.time()
        ediff <- scde.expression.difference(scde.fitted.model,combined.counts,scde.prior,groups=groups,n.cores=n.cores)
        print(proc.time() - t0)
        
        p.values <- 2*pnorm(abs(ediff$Z),lower.tail=F) # 2-tailed p-value
        p.values.adj <- 2*pnorm(abs(ediff$cZ),lower.tail=F) # Adjusted to control for FDR

        ## TODO: plot histogram of the raw p-values from this experiment.
        PlotPvalHist(p.values, paste(filename, ".png", sep=""))
        ##significant.genes <- which(p.values.adj<0.05)
        ##cat("\t", "num significant genes: ", length(significant.genes), "\n")
        
        ##ord <- order(p.values.adj[significant.genes]) # order by p-value
        ##de <- cbind(names(gene.symbols[significant.genes]),gene.symbols[significant.genes],ediff[significant.genes,1:3],p.values[significant.genes], p.values.adj[significant.genes])[ord,]
        ##colnames(de) <- c("EntrezID","Symbol", "Lower_bound","Log2_fold_change","Upper_bound","Raw_p_value", "Adj_p_value")

        de <- cbind(names(gene.symbols),gene.symbols,ediff[,1:3],p.values, p.values.adj)
        colnames(de) <- c("EntrezID", "Symbol", "Lower_bound", "Log2_fold_change", "Upper_bound", "Raw_p_value", "Adj_p_value")
        
        deg.results[[result.insert.idx]] <- de # Add the results of this study to the collection for this cell type
        result.insert.idx <- result.insert.idx + 1
        ## Also save the results of this experiment in its own file, to have on record

        filename <- paste(filename, ".csv", sep="")
        write.table(de, filename, sep=",", row.names=TRUE, col.names=TRUE)
    }
    
    ## Now, for this cell type, if we had more than one study, do a meta-analysis by taking maximum adjusted p-value as the p-value
    ## and the average of the log2(foldchange)
    if (length(deg.results) == 0) {
        cat("\t", "No experiments done for this type", "\n")
    } else {
        max.adj.pvals = integer(dim(counts.mat)[1])
        avg.fold.change = integer(dim(counts.mat)[1])
        for (results in deg.results) {
            max.adj.pvals <- pmax(max.adj.pvals, results[, "Adj_p_value"])
            avg.fold.change <- avg.fold.change + results[, "Log2_fold_change"]
        }
        avg.fold.change <- avg.fold.change / length(deg.results)

        ## Now, do significance and FDR adjustment using these new p-values
        ## TODO: should we re-adjust p-vals?
        ##meta.adj.pvals <- p.adjust(max.adj.pvals, method="BH")
        meta.adj.pvals <- max.adj.pvals
        sig.genes.sv<- meta.adj.pvals < 0.05
        cat("\t", "Num sig DEGs after meta analysis: ", sum(sig.genes.sv), "\n")

        ## Check consistency of Log2FoldChange sign accross experiments
        fold.change.signs = integer(dim(counts.mat)[1])
        for (results in deg.results) {
            fold.change.signs <- fold.change.signs + sign(results[, "Log2_fold_change"])
        }
        fold.change.consistent.sv <- abs(fold.change.signs) == length(deg.results)
        sig.and.consistent.sv <- sig.genes.sv & fold.change.consistent.sv
        cat("\t", "Num of sig DEGs with consistent foldchange accross experiments: ", sum(sig.and.consistent.sv), "\n")
        
        ord <- order(meta.adj.pvals[sig.and.consistent.sv]) # order by p-value
        final.results <- cbind(names(gene.symbols[sig.and.consistent.sv]), gene.symbols[sig.and.consistent.sv], avg.fold.change[sig.and.consistent.sv], meta.adj.pvals[sig.and.consistent.sv])[ord,]
        colnames(final.results) <- c("EntrezID", "Symbol", "Avg_log2_fold_change", "Max_adj_p_value")
        ## Finally, Save the results
        filename <- strsplit(type, split=" ")[[1]][1]
        filename <- paste(filename, "meta", sep="_")
        filename <- paste(filename, ".csv", sep="")
        write.table(final.results, filename, sep=",", row.names=TRUE, col.names=TRUE)
    }
    
}
