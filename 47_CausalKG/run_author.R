# This wrapper sets reproducible random draws and records the original result.
Sys.setenv(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
set.seed(20261029)
source("run/mediation_analysis.R", print.eval=TRUE)
result <- data.frame(n=nrow(data), acme_control=med_results$d0,
                     acme_treated=med_results$d1, acme_average=med_results$d.avg,
                     direct_effect_average=med_results$z.avg,
                     total_effect=med_results$tau.coef,
                     interaction_p=p_value_interaction,
                     fitted_noninteraction=exists("d.bin"))
stopifnot(all(is.finite(as.matrix(result[, 1:7]))))
write.csv(result, "run/result.csv", row.names=FALSE)
writeLines(capture.output(sessionInfo()), "run/R_session.txt")
