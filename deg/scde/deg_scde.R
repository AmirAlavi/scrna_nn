## Script to determine Differentially Expressed Genes (DEGs) for cell types
## using SCDE. Each cell type is a node in a "Cell Ontology", so "cell type"
## and "node" may be used interchangeably.

## "study", "accession", "experiment" are also sometimes used interchangeably.

## Order of the code in this file is most-specific -> high-level (bottom-up),
## thus it is recommended that you start reading at the bottom of the file


## Poorman's command line argument parsing:
args = commandArgs(trailingOnly=TRUE)
if (length(args) < 4) {
    stop("Missing args! arg1=data.dir, arg2=output.dir, arg3=n.cores, arg4=sample.size", call.=FALSE)
} else {
    data.dir <- args[1]
    output.dir <- args[2]
    n.cores <- strtoi(args[3])
    sample.size <- strtoi(args[4])
}

## *********************
## *** I/O functions ***
## *********************
WriteTable <- function(cell.type, table, extra.name = "") {
    ## Writes a results table out to a csv file
    filename <- strsplit(cell.type, split=" ")[[1]][1]
    filename <- paste(filename, extra.name, sep="")
    filename <- paste(filename, ".csv", sep="")
    write.table(table, filename, sep=",", row.names=TRUE, col.names=TRUE)
}

## ***************************
## *** Debugging functions ***
## ***************************
PrintHashMap <- function(hashmap) {
    # Print each key & value in 'hashmap' which is actually just an environment
    for (key in ls(hashmap)) {
        cat("key:", key, "\n")
        cat("value:")
        cat(str(get(key, envir = hashmap)), "\n\n")
    }
}

PrintSuitableNodes <- function(data.env, accessions.info.map) {
    ## Determine how many nodes we will be able to determine DEGs for with the given
    ## sample.size. Print these nodes.
    suitable.nodes.count <- 0
    cell.types.f <- factor(data.env$meta$labels)
    cat("Nodes for which we can determine DEGs:\n")
    for (type in levels(cell.types.f)) {
        accessions.info <- get(make.names(type), envir = accessions.info.map)
        suitable.accessions.sv <- accessions.info$num.cells >= data.env$sample.size
        if (sum(suitable.accessions.sv) >= 1 | sum(accessions.info$num.cells) >= data.env$sample.size) {
            suitable.nodes.count <- suitable.nodes.count + 1
            cat("\t", type, "\n")
        }
    }
    cat("Total: ", suitable.nodes.count, "\n\n")
}

## **************************
## *** Plotting functions ***
## **************************
PlotPvalHist <- function(pvals, filename) {
    png(filename)
    hist(pvals, breaks=100, col="grey", main=paste(filename, "DE raw p-vals", sep=" "), xlab="Raw p-values")
    dev.off()
}

## ***************************
## *** Filtering functions ***
## ***************************
CreateCellFilter <- function(data.env, threshold = 1.8e3) {
    ## Remove cells with fewer than 'threshold' detected genes (library size)
    csums <- colSums(data.env$counts.mat > 0)
    filter <- csums >= threshold
    return(filter)
}

CreateGeneFilter <- function(data.env, threshold = 50) {
    ## for each gene
    ##   for each cell type, calculate its average reads in that type. Cache it.
    ## Once you have calculated the within-type averages for a gene, get a final
    ## average over these averages for each gene. Then return a selection vector
    ## that only selects genes that are above a threshold on this metric.
    print("In CreateGeneFilter...")
    type.factor.vector <- factor(data.env$meta$labels)
    means.for.types <- list() # each element will be a dataframe that contains the means for a particular cell type
    insert.idx <- 1
    for(type in levels(type.factor.vector)){
        cur.type.selection.vector <- type.factor.vector == type
        cur.type.counts.mat <- data.env$counts.mat[, cur.type.selection.vector]
        means <- rowMeans(cur.type.counts.mat)
        means.for.types[[insert.idx]] <- means
        insert.idx <- insert.idx + 1
    }
    combined.mat <- do.call(cbind, means.for.types)
    final.means <- rowMeans(combined.mat)
    ## Take a cutoff on genes:
    filter.selection.vector <- final.means > threshold
    return(filter.selection.vector)
}

FilterCells <- function(data.env, to.keep.selection.vector) {
    ## Given a vector that selects cells, update data structures to keep only
    ## those cells, and remove the other cells.
    cat("Filtering out ", sum(!to.keep.selection.vector), " cells \n")
    data.env$counts.mat <- data.env$counts.mat[, to.keep.selection.vector]
    data.env$meta <- data.env$meta[to.keep.selection.vector, ]
}

FilterGenes <- function(data.env, to.keep.selection.vector) {
    ## Given a vector that selects genes, update data structures to keep only
    ## those genes, and remove the other genes.
    cat("Filtering out ", sum(!to.keep.selection.vector), " genes \n")
    data.env$counts.mat <- data.env$counts.mat[to.keep.selection.vector, ]
    data.env$gene.symbols <- data.env$gene.symbols[to.keep.selection.vector]
}

## ****************************
## *** Data setup functions ***
## ****************************
GetAccessionsInfoForCellType <- function(cell.type, cell.types.f, data.env) {
    ## For the given cell.type (cell.types.f is a factor vector of all cell types),
    ## create a dataframe with two columns:
    ##   The accession numbers that contain cells of this type
    ##   The numbers of this cell type that are contained in each of those accessions
    
    ## Factor vector
    accessions.f <- factor(data.env$meta$accessions)
    ## Selection vector
    cur.type.sv <- cell.types.f == cell.type
    
    cur.type.accessions <- unique(accessions.f[cur.type.sv])

    accns <- numeric()
    num.cells <- numeric()
    for(accn in cur.type.accessions) {
        cur.accn.sv <- accessions.f == accn
        cur.accn.and.cur.type.sv <- cur.type.sv & cur.accn.sv
        accns <- c(accns, accn)
        num.cells <- c(num.cells, sum(cur.accn.and.cur.type.sv))
    }
    return(data.frame(accns, num.cells))
}

GetAccessionsInfoMap <- function(data.env) {
    ## Return a hashmap whose keys are cell types and whose values are dataframes
    ## that contain information about the studies (accessions) for that cell type
    cell.type.accessions.map <- new.env(parent = emptyenv())
    ## Factor vector
    cell.types.f <- factor(data.env$meta$labels)
    for (type in levels(cell.types.f)) {
        type.as.name <- make.names(type)
        cell.type.accessions.map[[type.as.name]] <- GetAccessionsInfoForCellType(type, cell.types.f, data.env)
    }
    return(cell.type.accessions.map)
}

LoadData <- function(data.dir) {
    ## Load R objects containing data into a separate environment that acts as a container
    ## for data. Return this environment.
    cat("Looking in ", data.dir, "\n") ### Must have a data.dir variable in workspace!

    data.env <- new.env(parent = emptyenv())
    load(paste(data.dir, "/", "counts_mat.gzip", sep=""), envir = data.env)   # counts.mat
    load(paste(data.dir, "/", "accessions.gzip", sep=""), envir = data.env)   # accessions
    load(paste(data.dir, "/", "gene_symbols.gzip", sep=""), envir = data.env) # gene.symbols
    load(paste(data.dir, "/", "labels.gzip", sep=""), envir = data.env)       # labels

    ## Some preprocessing to get in the right form
    data.env$counts.mat <- data.matrix(data.env$counts.mat, rownames.force=TRUE)
    data.env$counts.mat <- t(data.env$counts.mat) ### Most DE tools expect rows to be genes, columns to be samples

    data.env$meta <- data.frame(labels = data.env$labels, accessions = data.env$accessions)
    rm(accessions, envir = data.env)
    rm(labels, envir = data.env)
    print("Done loading data objects")
    return(data.env)
}

## *********************************************************
## *** Differential expression code (in bottom-up order) ***
## *********************************************************
CalcStats <- function(de.results, data.env) {
    ## Finally, compute p-values
    p.values <- 2*pnorm(abs(de.results$Z),lower.tail=F) # 2-tailed p-value
    p.values.adj <- 2*pnorm(abs(de.results$cZ),lower.tail=F) # Adjusted to control for FDR
    processed.results <- cbind(names(data.env$gene.symbols), data.env$gene.symbols, de.results[,1:3], p.values, p.values.adj)
    colnames(processed.results) <- c("EntrezID", "Symbol", "Lower_bound", "Log2_fold_change", "Upper_bound", "Raw_p_value", "Adj_p_value")
    return(processed.results)
}

RunSCDE <- function(counts, groups, data.env) {
    ## Call SCDE functions to compute differential expression
    cat("\t", "\t", "Fitting error models...", "\n")
    t0 <- proc.time()
    scde.fitted.model <- knn.error.models(counts = counts, groups = groups, n.cores = data.env$n.cores, save.model.plots = F)
    print(proc.time() - t0)
    scde.prior <- scde.expression.prior(models = scde.fitted.model, counts = counts)
    
    cat("\t", "\t", "Calculating differential expression...", "\n")
    t0 <- proc.time()
    ediff <- scde.expression.difference(scde.fitted.model, counts, scde.prior, groups = groups, n.cores = data.env$n.cores)
    print(proc.time() - t0)
    processed.results <- CalcStats(ediff, data.env)
    cat("dim(processed.results): ", dim(processed.results), "\n")
    return(processed.results)
}

SetUpAndRunSCDE <- function(cell.type, cell.type.sv, other.types.sv, data.env) {
    ## Given a cell.type, and a selection vector that selects cells of that type from
    ## a larger counts matrices (and the same for other cell types),
    ## setup the counts matrices and group labels for SCDE
    cell.type.counts <- data.env$counts.mat[, cell.type.sv]
    other.types.counts <- data.env$counts.mat[, other.types.sv]
    ## Take a sample from these matrices
    cell.type.counts <- cell.type.counts[, sample(ncol(cell.type.counts), data.env$sample.size)]
    other.types.counts <- other.types.counts[, sample(ncol(other.types.counts), data.env$sample.size)]
    
    combined.counts <- cbind(other.types.counts, cell.type.counts)
    grouping <- rep(c("OTHERS", make.names(cell.type)), each = data.env$sample.size)
    groups <- factor(grouping)
    
    results <- RunSCDE(combined.counts, groups, data.env)
    return(results)
}

MultipleDegExperiments <- function(cell.type, accns, data.env) {
    ## Do DEG experiments on a study-by-study basis (each accn is a separate
    ## experiment for the given cell.type). Add the results from each of these into
    ## a list which is returned
    deg.results <- list()
    deg.results.idx <- 1
    for (accn in accns) {
        cat("\t\t", "Doing study: ", accn, "\n")
        ## Selection vectors
        cell.type.and.accn.sv <- data.env$meta$labels == cell.type & data.env$meta$accessions == accn
        other.types.sv <- data.env$meta$labels != cell.type
        results <- SetUpAndRunSCDE(cell.type, cell.type.and.accn.sv, other.types.sv, data.env)
        ## save results
        WriteTable(cell.type, results, extra.name = paste("_", accn, sep=""))
        deg.results[[deg.results.idx]] <- results
        deg.results.idx <- deg.results.idx + 1
        cat("\n")
    }
    return(deg.results)
}

SingleDegExperiment <- function(cell.type, data.env) {
    ## Do a single DEG experiment for this type (because there were not enough cells in
    ## separate studies for this cell.type, so we combine the cells from all studies of
    ## this cell.type into one DEG experiment).
    cell.type.sv <- data.env$meta$labels == cell.type
    other.types.sv <- !cell.type.sv
    result <- SetUpAndRunSCDE(cell.type, cell.type.sv, other.types.sv, data.env)
    ## save results
    WriteTable(cell.type, result, extra.name = "_combined")
    return(list(result))
}

DegForCellType <- function(cell.type, data.env, accessions.info.map) {
    ## Do the DEG experiment(s) for this cell type.
    accessions.info <- get(make.names(cell.type), envir = accessions.info.map)
    suitable.accessions.sv <- accessions.info$num.cells >= data.env$sample.size
    if (sum(suitable.accessions.sv) >= 1) {
        ## If there exists at least one study (accn) that contains the sample.size number
        ## of cells of this type, then do meta-experiment that combines results from separate
        ## experiments for each suitably-sized accession
        cat("\t", "Doing studies in separate experiments...\n")
        accns <- accessions.info$accns[suitable.accessions.sv]
        return(MultipleDegExperiments(cell.type, accns, data.env))
    } else if (sum(accessions.info$num.cells) >= data.env$sample.size) {
        ## Otherwise, if we can get sample.size number of cells by combining all samples of
        ## this cell from all of its studies, do single experiment with all samples
        cat("\t", "Doing a combined experiment...\n")
        return(SingleDegExperiment(cell.type, data.env))
    } else {
        cat("\t", "Too few samples for ", cell.type, ", skipping cell type", "\n")
        return(list())
    }
}


RunDegExperiments <- function(data.env, accessions.info.map) {
    cell.types.f <- factor(data.env$meta$labels)
    for (type in levels(cell.types.f)) {
        cat("Current type: ", type, "\n")
        cell.type.deg.results <- DegForCellType(type, data.env, accessions.info.map)
        if (length(cell.type.deg.results)) {
            meta.deg.results <- AnalyzeDegResults(cell.type.deg.results, data.env)
            WriteTable(type, meta.deg.results, extra.name = "_meta")
        }
    }
}

## ****************************************
## *** Meta-analysis functions ***
## ****************************************
AnalyzeDegResults <- function(deg.results, data.env) {
    ## Do a meta-analaysis that combines the results of multiple DEG experiments
    ## for a cell.type into a single list of DEGs with their p-values and log
    ## fold changes.
    max.adj.pvals = integer(dim(data.env$counts.mat)[1])
    avg.fold.change = integer(dim(data.env$counts.mat)[1])
    for (results in deg.results) {
        max.adj.pvals <- pmax(max.adj.pvals, results[, "Adj_p_value"])
        avg.fold.change <- avg.fold.change + results[, "Log2_fold_change"]
    }
    avg.fold.change <- avg.fold.change / length(deg.results)

    ## Now, do significance and FDR adjustment using these new p-values
    ## TODO: should we re-adjust p-vals?
    ##meta.adj.pvals <- p.adjust(max.adj.pvals, method="BH")
    meta.adj.pvals <- max.adj.pvals
    sig.genes.sv <- meta.adj.pvals < 0.05
    cat("\t", "Num sig DEGs after meta analysis: ", sum(sig.genes.sv), "\n")

    ## Check consistency of Log2FoldChange sign accross experiments
    fold.change.signs = integer(dim(data.env$counts.mat)[1])
    for (results in deg.results) {
        fold.change.signs <- fold.change.signs + sign(results[, "Log2_fold_change"])
    }
    fold.change.consistent.sv <- abs(fold.change.signs) == length(deg.results)
    sig.and.consistent.sv <- sig.genes.sv & fold.change.consistent.sv
    cat("\t", "Num of sig DEGs with consistent foldchange accross experiments: ", sum(sig.and.consistent.sv), "\n")
        
    ord <- order(meta.adj.pvals[sig.and.consistent.sv]) # order by p-value
    entrez.IDs <- names(data.env$gene.symbols[sig.and.consistent.sv])
    gene.symbols <- data.env$gene.symbols[sig.and.consistent.sv]
    final.results <- cbind(entrez.IDs, gene.symbols, avg.fold.change[sig.and.consistent.sv], meta.adj.pvals[sig.and.consistent.sv])[ord,]
    colnames(final.results) <- c("EntrezID", "Symbol", "Avg_log2_fold_change", "Max_adj_p_value")
    return(final.results)
}

### ***************************
### *** Script-driving code ***
### ***************************
library(scde)

cat("Using ", n.cores, " cores", "\n")
data.env <- LoadData(data.dir)
## Also place program params in data hashmap for convenience
data.env$n.cores <- n.cores
data.env$sample.size <- sample.size
cat("dim(data.env$counts.mat) (Before filtering): ", dim(data.env$counts.mat), "\n")

dir.create(output.dir, recursive = TRUE)
setwd(output.dir)

print("Filtering...")
gene.thresh.selection.vector <- CreateGeneFilter(data.env) 
FilterGenes(data.env, gene.thresh.selection.vector)

cells.thresh.selection.vector <- CreateCellFilter(data.env)
FilterCells(data.env, cells.thresh.selection.vector)
cat("dim(data.env$counts.mat) (After filtering): ", dim(data.env$counts.mat), "\n")

accessions.info.map <- GetAccessionsInfoMap(data.env)
## PrintHashMap(accessions.info.map)
## PrintSuitableNodes(data.env, accessions.info.map)
RunDegExperiments(data.env, accessions.info.map)
