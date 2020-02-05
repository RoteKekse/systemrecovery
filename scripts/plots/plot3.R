require(tikzDevice)

s1 = "RANDOM_FPTU_MEAN_rals_model_b_test_2_results"
s2 = "RANDOM_FPTU_MEAN_rals_model_b_test_3_results"

linewidth = 3
c1 = "#CC3333"
c2 = "#339900"
c3 = "#990066"

data <- read.csv(paste(s1,'.csv', sep=''))
data2 <- read.csv(paste(s2,'.csv', sep=''))
data
data2
tikz(paste('tikz/',s1,'.tex', sep=''), width=5, height=3)
#tikz(paste('tikz/',s1,'.tex', sep=''), standAlone=TRUE, width=5, height=3)
plot(data[1:8,2],data[1:8,3],type="b",col=c1,lwd=linewidth,
     xlab="Number of observations", ylab="Recovery rate", ylim=c(0,1), xaxt='n',yaxt='n',xlim=c(0,12000),lty=1,pch=17,cex = 0.4)
axis(side = 2, at = c(0.0,0.5,1.0),labels = T)
axis(side = 1, at = c(0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000),labels = T)

lines(data[9:16,2],data[9:16,3],col=c2,lwd=linewidth,lty=1,type="b",pch=18,cex = 0.4)
x18 = c(data[17:24,2], data2[1:5,2])
y18 = c(data[17:24,3] , data2[1:5,3])
lines(x18,y18,col=c3,lwd=linewidth,lty=1,type="b",pch=19,cex = 0.4)

legend(10000,0.4, legend=c("d = 6", "d = 12", "d = 18"),col=c(c1,c2,c3),cex=0.8,lty=1,lwd=linewidth,pch=c(17,18,19))

dev.off()
#tools::texi2pdf(paste('tikz/',s1,'.tex', sep=''))
#system(paste(getOption('pdfviewer'),paste(s1,'.pdf', sep='')))
