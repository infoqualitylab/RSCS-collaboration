# Mark Van Moer NCSA/RSCS/UIUC
# make boxplots of years of publication of cited articles

library(tidyverse)

projectpath <- "~/Vis/projects/schneider-rscs/"
setwd(projectpath)

# Salt
saltdatapath <- str_c(projectpath, "data/original/databank/")
saltnodesfile <- str_c(saltdatapath, "article_list.csv")
saltedgesfile <- str_c(saltdatapath, "inclusion_net_edges.csv")

saltnodes <- read_csv(saltnodesfile)
saltedges <- read_csv(saltedgesfile)
saltedges <- saltedges %>% filter(!is.na(cited_ID))

tmp <- left_join(saltedges, saltnodes, by=c("citing_ID" = "ID"))
salt <- left_join(tmp, saltnodes, by=c("cited_ID" = "ID"))

psalt <- salt %>%
  ggplot() +
  geom_boxplot(aes(fct_reorder2(short_name.x,short_name.x,year.x, .desc=TRUE), year.y),varwidth = TRUE)+
  coord_flip() +
  labs(title="Salt",
       subtitle="distribution of citation year",
       x="Review Article",
       y="Year")
ggsave('./Salt_boxplot_citation_year_thickness.png', psalt, width=8.5, height=11, units='in')

# ExRx
exrxdatapath <- str_c(projectpath, "data/original/exrx/")
exrxnodesfile <- str_c(exrxdatapath, "Article_list_June_11.csv")
exrxedgesfile <- str_c(exrxdatapath, "edited_inclusion_net_edges_June_11.csv")

exrxnodes <- read_csv(exrxnodesfile)
exrxedges <- read_csv(exrxedgesfile)
exrxedges <- exrxedges %>% filter(!is.na(cited_ID))

# ExRx renaming cols
exrxnodes <- exrxnodes %>% rename(
   ID = `Our ID`,
   year = `Publication Year`
)

# create short names for ExRx
# not perfect, e.g. Kitman HE; ... fails
exrxnodes <- exrxnodes %>% rowwise() %>%
    mutate(
      short_name = str_c(
        str_split(
          str_split(Author, ';')[[1]][1], ',')[[1]][1],
        year)
    )

tmp <- left_join(exrxedges, exrxnodes, by=c("citing_ID" = "ID"))
exrx <- left_join(tmp, exrxnodes, by=c("cited_ID" = "ID"))
pexrx <- exrx %>%
  ggplot() +
  geom_boxplot(aes(fct_reorder2(short_name.x,short_name.x,year.x, .desc=TRUE), year.y),varwidth = TRUE)+
  coord_flip() +
  labs(title="ExRx",
       subtitle="distribution of citation year",
       x="Review Article",
       y="Year")
ggsave('./ExRx_boxplot_citation_year_thickness.png', pexrx, width=8.5, height=11, units='in')