
library(R.matlab)
library(xgboost)
library(pROC)
PATH <- getwd();
setwd(PATH)

# 定义样本数量
Trainnum <- 10000
drow <- 145  # 行
dcol <- 145  # 列
ddim1 <- 220 # 波段

pathname1 <- file.path('Indian','Indian_pines.mat')
pathname2 <- file.path('Indian','Indian_pines_gt.mat')
Indian_fea <- readMat(pathname1) # 将filename文件的内容以mat形式读出
Indian_gt <- readMat(pathname2)

# 将list 转为矩阵
Indian_fea <- array(do.call(cbind, Indian_fea),dim = c(21025,ddim1))#145*145
Indian_gt <- array(do.call(cbind, Indian_gt ),dim = c(21025,1))

# 将特征和样本特征合并在一个数组中 221=220+1
Indian_data <- array(data = cbind(Indian_fea,Indian_gt),dim = c(21025,221))

# 去除gt中的0元素
Indian_gtIndex <- which(Indian_gt>0)# 返回非零元素的下标
Indian_data_no0 <- array(data =NA,dim = c(length(Indian_gtIndex),221))
Indian_data_no0 <- Indian_data[Indian_gtIndex,]

# 获取训练集
set.seed(30)
x = 1:length(Indian_data_no0[,221])
RandomIndex <- sample(x=x,length(x))
TrainIndex <- RandomIndex[1:Trainnum]
TestIndex <- array(data = RandomIndex[Trainnum+1:length(x)],dim=c(1,length(x)-Trainnum))

TrainSets <- array(data = NA,dim = c(Trainnum,221))
TrainSets <- Indian_data_no0[TrainIndex,]
TestSets <- array(data = NA,dim = c(length(x)-Trainnum,221))
TestSets <- Indian_data_no0[TestIndex,]

# xgboost模型
Indian_fea1 <- array(Indian_fea, dim = c(21025,Trainnum))
dtest <- xgb.DMatrix(data = TestSets[,1:220], label = TestSets[,221]-2)
dtrain <- xgb.DMatrix(data = TrainSets[,1:220], label = TrainSets[,221]-2)
xgb <- xgboost(data = dtrain,max_depth=6, eta=0.5,  nround=25)

#在测试集上预测 
pre_xgb <- round(predict(xgb,newdata = dtest))+2 

Ma <- max(pre_xgb)
Mi <- min(pre_xgb)

#还原预测图象
Indian_preno0 <- array(data = 0, dim = c(length(x),1))
Indian_preno0[TestIndex] <- pre_xgb
Indian_preno0[TrainIndex]<- TrainSets[,221]
Indian_pre <- array(data = 0,dim = c(21025,1))
Indian_pre[Indian_gtIndex] = Indian_preno0
pred <- array(data = Indian_pre, dim = c(145,145))

# 设置图像显示颜色
par(mfrow=c(1,1))
par(mar=c(0.1,0.1,2,0.1), xaxs="i", yaxs="i")
mycolors <- c(terrain.colors(16)[c(16)],
              rainbow(16)[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)])
barplot(rep(1,times=10),col=mycolors,border=mycolors,axes=FALSE,
        main="myColor"); box()

# 显示预测图像
image(pred, col = mycolors)
