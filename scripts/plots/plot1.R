require(tikzDevice)

s1 = "FPTU_salsa_model_a_test_4_results"
s2 = "FPTU_rals_model_b_test_6_results"
s3 = "FPTU_salsa_model_a_test_5_results"

data <- read.csv(paste(s1,'.csv', sep=''))
data2 <- read.csv(paste(s2,'.csv', sep=''))
data3 <- read.csv(paste(s3,'.csv', sep=''))

linewidth = 3
c1 = "#CC3333"
c2 = "#339900"
c3 = "#990066"
black = "#000000"



tikz(paste('tikz/',s1,'.tex', sep=''), width=5, height=3)
#tikz(paste('tikz/',s1,'.tex', sep=''), standAlone=TRUE, width=5, height=3)
plot(data[1:5,2],data[1:5,3],type="b",col=c1,lwd=linewidth,
     xlab="Number of observations", ylab="Recovery rate", ylim=c(0,1), xaxt='n',yaxt='n',xlim=c(0,6000),lty=3,pch=17,cex = 0.7)
axis(side = 2, at = c(0.0,0.5,1.0),labels = T)
axis(side = 1, at = c(0,1000,2000,3000,4000,5000,6000),labels = T)
x12 = c(data[8:9,2], data3[,2])
y12 = c(data[8:9,3] , data3[,3])
lines(x12,y12,col=c2,lwd=linewidth,lty=3,type="b",pch=18,cex = 0.9)

lines(data2[1:8,2],data2[1:8,3],col=c1,lwd=linewidth,lty=5,type="b",pch=17,cex = 0.7)
lines(data2[9:16,2],data2[9:16,3],col=c2,lwd=linewidth,lty=5,type="b",pch=18,cex = 0.9)
lines(data2[17:24,2],data2[17:24,3],col=c3,lwd=linewidth,lty=5,type="b",pch=19,cex = 0.7)

legend(4200,0.54, legend=c("d = 6", "d = 12", "d = 18","ALS+selection",'SALSA+singleTT'),col=c(c1,c2,c3,black,black),cex=0.8,lty=c(1,1,1,5,3),lwd=linewidth,pch=c(17,18,19,-1,-1))

dev.off()
#tools::texi2pdf(paste('tikz/',s1,'.tex', sep=''))
#system(paste(getOption('pdfviewer'),paste(s1,'.pdf', sep='')))
