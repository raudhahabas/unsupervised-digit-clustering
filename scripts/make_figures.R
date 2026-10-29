# Generates the PNG figures used in README.md.
# Run from the repository root:  Rscript scripts/make_figures.R

library(ggplot2)

dir.create("figures", showWarnings = FALSE)

# Colour-blind-safe palette (one colour per digit) and neutral inks
digit_cols <- c("5" = "#2a78d6", "6" = "#eb6834", "7" = "#1baf7a")
ink        <- "#0b0b0b"
ink_muted  <- "#52514e"
grid_col   <- "#e6e5e1"

theme_clean <- theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = grid_col, linewidth = 0.4),
        plot.title = element_text(face = "bold", colour = ink),
        plot.subtitle = element_text(colour = ink_muted),
        axis.title = element_text(colour = ink_muted),
        axis.text = element_text(colour = ink_muted),
        legend.position = "top",
        legend.justification = "left",
        plot.background = element_rect(fill = "white", colour = NA))

# ---- Data and PCA ----------------------------------------------------------
read_mat  <- as.matrix(read.table("data/digit.txt", sep = ","))
image_mat <- read_mat[, 2:785]
label     <- read_mat[, 1]
true_label <- as.numeric(factor(label))

pca_out <- prcomp(image_mat)
pca20   <- pca_out$x[, 1:20]

# ---- Clustering (same settings as the report) -------------------------------
perms <- list(c(1,2,3), c(1,3,2), c(2,1,3), c(2,3,1), c(3,1,2), c(3,2,1))
match_clusters <- function(cl) {
  errors <- sapply(perms, function(p) sum(p[cl] != true_label))
  perms[[which.min(errors)]][cl]
}

set.seed(1)
km_cluster <- kmeans(pca20, centers = 3, nstart = 25)$cluster
dmat <- dist(pca20)
cuts <- lapply(c(single = "single", complete = "complete",
                 average = "average", ward = "ward.D2"),
               function(m) cutree(hclust(dmat, method = m), k = 3))

accuracy <- function(cl) mean(match_clusters(cl) == true_label)

# ---- 1. Hero: original digits vs PCA reconstructions -----------------------
mu <- colMeans(image_mat)
U  <- pca_out$rotation
reconstruct <- function(i, K) {
  if (K == 0) return(image_mat[i, ])
  drop(pca_out$x[i, 1:K, drop = FALSE] %*% t(U[, 1:K, drop = FALSE])) + mu
}

set.seed(256)
idx <- c(sample(which(label == 5), 1),
         sample(which(label == 6), 1),
         sample(which(label == 7), 1))
Ks  <- c(1, 2, 5, 10, 20, 50, 0)
col_names <- c(paste(Ks[-length(Ks)], ifelse(Ks[-length(Ks)] == 1, "PC", "PCs")),
               "Original (784 px)")

tiles <- do.call(rbind, lapply(seq_along(idx), function(r) {
  do.call(rbind, lapply(seq_along(Ks), function(c) {
    m <- matrix(reconstruct(idx[r], Ks[c]), nrow = 28)[, 28:1]
    data.frame(x = rep(1:28, 28), y = rep(1:28, each = 28),
               value = as.vector(m),
               digit = paste("Digit", label[idx[r]]),
               k = factor(col_names[c], levels = col_names))
  }))
}))

p_hero <- ggplot(tiles, aes(x, y, fill = value)) +
  geom_raster() +
  scale_fill_gradient(low = "white", high = ink, guide = "none",
                      limits = c(0, max(image_mat)), oob = scales::squish) +
  facet_grid(digit ~ k, switch = "y") +
  coord_equal() +
  labs(title = "Rebuilding handwritten digits from a few principal components",
       subtitle = "Each image starts as 784 pixel values. Around 10 to 20 components are enough to recognise the digit.") +
  theme_void(base_size = 13) +
  theme(plot.title = element_text(face = "bold", colour = ink, margin = margin(b = 4)),
        plot.subtitle = element_text(colour = ink_muted, margin = margin(b = 10)),
        strip.text.x = element_text(colour = ink_muted, margin = margin(b = 6)),
        strip.text.y.left = element_text(colour = ink, face = "bold", angle = 0,
                                         margin = margin(r = 8)),
        panel.spacing = unit(6, "pt"),
        panel.border = element_rect(colour = grid_col, fill = NA, linewidth = 0.5),
        plot.margin = margin(14, 14, 14, 14),
        plot.background = element_rect(fill = "white", colour = NA))
ggsave("figures/hero_reconstruction.png", p_hero, width = 11, height = 5.4, dpi = 160)

# ---- 2. PCA map coloured by true digit -------------------------------------
pc_df <- data.frame(PC1 = pca20[, 1], PC2 = pca20[, 2], digit = factor(label))
p_pca <- ggplot(pc_df, aes(PC1, PC2, colour = digit)) +
  geom_point(size = 1.4, alpha = 0.55) +
  scale_colour_manual(values = digit_cols, name = "True digit") +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  labs(title = "Digits separate on just two principal components",
       subtitle = "Labels are used only to colour the points, never to fit the model") +
  theme_clean
ggsave("figures/pca_scatter.png", p_pca, width = 7.5, height = 5.6, dpi = 160)

# ---- 3. Variance explained -------------------------------------------------
var_exp <- pca_out$sdev^2 / sum(pca_out$sdev^2)
cum_df  <- data.frame(k = 1:100, cum = cumsum(var_exp)[1:100])
cum20   <- cum_df$cum[20]
p_var <- ggplot(cum_df, aes(k, cum)) +
  geom_line(colour = digit_cols[["5"]], linewidth = 1) +
  geom_vline(xintercept = 20, colour = ink_muted, linetype = "dashed", linewidth = 0.4) +
  annotate("point", x = 20, y = cum20, colour = digit_cols[["5"]], size = 3) +
  annotate("text", x = 22, y = cum20 - 0.04, hjust = 0, colour = ink,
           label = sprintf("20 components keep %.0f%% of the variance", 100 * cum20)) +
  scale_y_continuous(labels = function(v) paste0(round(100 * v), "%"),
                     limits = c(0, 1)) +
  labs(title = "Most of the variation lives in a few directions",
       subtitle = "Cumulative share of pixel variance explained by the first k components",
       x = "Number of principal components (k)", y = NULL) +
  theme_clean
ggsave("figures/variance_explained.png", p_var, width = 7.5, height = 4.6, dpi = 160)

# ---- 4. Clustering accuracy by method -------------------------------------
acc_df <- data.frame(
  method = c("K-means", "Single linkage", "Complete linkage",
             "Average linkage", "Ward's linkage"),
  acc = c(accuracy(km_cluster), accuracy(cuts$single), accuracy(cuts$complete),
          accuracy(cuts$average), accuracy(cuts$ward)))
acc_df$method <- factor(acc_df$method, levels = acc_df$method[order(acc_df$acc)])
acc_df$best   <- acc_df$acc == max(acc_df$acc)

p_acc <- ggplot(acc_df, aes(acc, method, fill = best)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * acc)), hjust = -0.15,
            colour = ink, size = 4.2) +
  geom_vline(xintercept = 1/3, colour = ink_muted, linetype = "dashed", linewidth = 0.4) +
  annotate("text", x = 1/3, y = 5.45, label = "random guess (33%)",
           hjust = -0.05, colour = ink_muted, size = 3.4) +
  scale_fill_manual(values = c(`TRUE` = digit_cols[["5"]], `FALSE` = "#b9b8b2"),
                    guide = "none") +
  scale_x_continuous(labels = function(v) paste0(round(100 * v), "%"),
                     limits = c(0, 1.1), breaks = seq(0, 1, 0.25)) +
  labs(title = "Ward's method recovers the digits best",
       subtitle = "Share of 3,000 images put in the right group (no labels used)",
       x = NULL, y = NULL) +
  theme_clean +
  theme(panel.grid.major.y = element_blank())
ggsave("figures/clustering_accuracy.png", p_acc, width = 7.5, height = 4.2, dpi = 160)

# ---- 5. Ward clusters with errors highlighted ------------------------------
ward_matched <- match_clusters(cuts$ward)
ward_df <- data.frame(PC1 = pca20[, 1], PC2 = pca20[, 2],
                      cluster = factor(c("5", "6", "7")[ward_matched]),
                      wrong = ward_matched != true_label)
p_ward <- ggplot(ward_df, aes(PC1, PC2)) +
  geom_point(aes(colour = cluster), size = 1.3, alpha = 0.35) +
  geom_point(data = subset(ward_df, wrong), shape = 21, size = 2.6,
             stroke = 0.9, colour = ink, fill = NA) +
  scale_colour_manual(values = digit_cols, name = "Ward cluster (matched digit)") +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  labs(title = sprintf("Only %d of 3,000 images end up in the wrong cluster",
                       sum(ward_df$wrong)),
       subtitle = "Black rings mark misclassified images; most are 5s that were grouped with the 6s") +
  theme_clean
ggsave("figures/ward_misclassified.png", p_ward, width = 7.5, height = 5.6, dpi = 160)

print(acc_df)
cat(sprintf("Variance kept by 20 PCs: %.1f%%\n", 100 * cum20))
print(table(True_Digit = label, Ward = c(5, 6, 7)[ward_matched]))
