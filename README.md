## "Titanic - Parte 1"
Autor: "Lucas Tavares"


```{r message=FALSE}
library(dplyr)
library(ggplot2)
library(gridExtra)
library(stringr)
library(randomForest)
library(e1071)
library(rlist)
```

## Importação dos dados

Primeiramente precisamos importar os arquivos .csv.

```{r}
titanic_train = read.csv("C:/Users/Lucas/Desktop/Organização/Data Science/Titanic/train.csv", header=TRUE)

titanic_test = read.csv("C:/Users/Lucas/Desktop/Organização/Data Science/Titanic/test.csv", header=TRUE)
```

## Conhecendo e otimizando os dados

Ao observarmos as variáveis com a função head() percebemos algumas questões interessantes.

```{r echo=FALSE}
head(titanic_train)
```

Primeiramente, os valores ausentes da coluna Cabin não estão computados como NA, o que evitará que esses valores sejam encontrados em uma futura busca.

Além disso, percebemos que, apesar de não utilizarmos a variável Name, podemos utilizar os prefixos (Mr, Mrs, etc). 

Sendo assim, precisamos resolver essas questões.

```{r}
# Passar valores ausentes para NA.

titanic_train$Cabin[titanic_train$Cabin==""]=NA

# Criar uma nova variável "prefix" com os prefixos a partir da variável name.

titanic_train = titanic_train %>% mutate(Prefix = as.factor(str_sub(Name, regexpr(",", Name)+2,regexpr("\\.", Name)-1)))
```
```{r echo=FALSE}
head(titanic_train)
```


## Preparo dos dados

Aqui precisamos preparar os dados para futuras análises. O primeiro passo é encontrar dados ausentes (NAs).


```{r}
hNA = sapply(titanic_train, function(x) sum(is.na(x)))
names = names(hNA)
hNA = as.data.frame(hNA) %>% mutate(percent=hNA/nrow(titanic_train)*100)
row.names(hNA)=names
hNA[!hNA$percent==0,]
```

As variáveis Age (Idade) e Cabin (Cabine) possuem valores ausentes que correspondem a 19.8% e 77.1% da amostra total respectivamente.


## Seleção de variáveis

Agora, precisamos selecionar quais variáveis entrarão no modelo. O primeiro passo é analisar as variáveis categóricas (fatores) e numéricas separadamente.

```{r}
# Primeiro criamos objetos iguais ao dataset para cada tipo de variável.

train_factor=titanic_train
train_num=titanic_train

# Agora executamos um loop que excluirá variáveis de tipos diferentes aqueles correspondentes.

for(c in names(titanic_train))
{
  if(is.factor(titanic_train[,(c)])==FALSE)
  {train_factor[(c)]=NULL} 
  else if(is.numeric(titanic_train[,(c)])==FALSE)
  {train_num[(c)]=NULL}
}

cat("Fatores:", paste(names(train_factor), collapse = ", "),"\n")
cat("Numéricos:", paste(names(train_num),collapse = ", "))
```

Agora precisamos saber o quanto cada variável influencia na variável survived.
Para isso vamos criar alguns gráficos.

```{r}
# Começamos por variáveis categóricas

gsex = titanic_train %>% ggplot(aes(Sex)) +
  geom_bar(aes(color = as.factor(Survived), fill = as.factor(Survived)))
gprefix = titanic_train %>% ggplot(aes(Prefix)) +
  geom_bar(aes(color = as.factor(Survived), fill = as.factor(Survived)))
gpclass = titanic_train %>% ggplot(aes(Pclass)) +
  geom_bar(aes(color = as.factor(Survived), fill = as.factor(Survived)))
gticket = titanic_train %>% ggplot(aes(Ticket)) +
  geom_bar(aes(color = as.factor(Survived), fill = as.factor(Survived)))
gcabin = titanic_train %>% ggplot(aes(Cabin)) +
  geom_bar(aes(color = as.factor(Survived), fill = as.factor(Survived)))
gembarked = titanic_train %>% ggplot(aes(Embarked)) +
  geom_bar(aes(color = as.factor(Survived), fill = as.factor(Survived)))

grid.arrange(gsex, gprefix, gpclass, gembarked, gcabin, gticket, nrow=3)
```

Como podemos observar, a distribuição de sobreviventes e não sobreviventes é bastante diferente em categorias nas variáveis Sex, Prefix, Pclass e Embarked.
Sendo assim, essas categorias podem ser importantes para o modelo.

As variáveis Ticket e Cabin não demonstram alguma relação com a classe (Survived), além de existirem muitos valores ausentes (77.1%) em Cabin.

Sendo assim, as variáveis Cabin e Ticket não serão utilizadas no modelo.

```{r}
# Agora analisaremos as variáveis numéricas.

gage = titanic_train %>% ggplot(aes(Age)) +
  geom_histogram(aes(color = as.factor(Survived), fill = as.factor(Survived)), binwidth = 5)
gsibsp = titanic_train %>% ggplot(aes(SibSp)) +
  geom_histogram(aes(color = as.factor(Survived), fill = as.factor(Survived)), binwidth = 1)
gparch = titanic_train %>% ggplot(aes(Parch)) +
  geom_histogram(aes(color = as.factor(Survived), fill = as.factor(Survived)), binwidth = 1)
gfare = titanic_train %>% ggplot(aes(Fare)) +
  geom_histogram(aes(color = as.factor(Survived), fill = as.factor(Survived)), binwidth = 30)

grid.arrange(gage, gsibsp, gparch, gfare, nrow=2)
```

Aqui percebemos que todas as variáveis parecem ter uma boa relação com Survived. 

- Para Age, ter entre 20 e 40 anos de idade parece ser desvaforável à sobrevivência. 

- Em SibSp e Parch, não ter nenhum familiar a bordo parece ser também um fator desfavorável. 

- Em Fare por fim, pessoas que não pagaram para estar a bordo e pessoas que pagaram até 50$ tiveram uma taxa bem grande de mortes, o que mostra uma relação importante com a sobrevivência dessas pessoas.

Sendo assim, todas as variáveis numéricas serão utilizadas no modelo. O próximo passo será preencher os valores ausentes na variável Age através da predição por regressão linear.

* Note que esse código gera uma um aviso de exclusão dos 177 valores que estão ausentes em Age.

## Predição de valores ausentes em Age nos dados de treino

Para predizer os valores ausentes em Age utilizaremos a técnica de regressão linear.

```{r}
# Preparo dos dados

dadosrl = titanic_train %>% 
  mutate(PassengerId=NULL, Name=NULL, Ticket = NULL, Cabin=NULL, Survived=NULL) # Precisamos retirar as variáveis PassengerID, Name, Ticket e Cabin que não serão úteis para a predição.

modelorl = lm(dadosrl$Age ~ .,dadosrl)

# Agora fazemos a predição dos dados de treino.

  modelorl$xlevels[["Prefix"]] <- union(modelorl$xlevels[["Prefix"]], levels(titanic_train[["Prefix"]]))
  modelorl$xlevels[["Embarked"]] <- union(modelorl$xlevels[["Embarked"]], levels(titanic_train[["Embarked"]]))
    
  predrl = predict(modelorl, titanic_train)
    
# Por fim substituímos os valores ausentes pelos valores preditos.
    
    for (r in 1:nrow(titanic_train)){if(is.na(titanic_train[r,("Age")]==TRUE))
    {titanic_train[r,("Age")]=predrl[r]}}
    
```

## Estratificação treino-teste

Aqui precisamos separar uma amostra para treino e outra para teste aleatóriamente.

```{r}
# Primeiro criamos uma amostra aleatória de valores binários (0 e 1).
amostra=sample(c(0,1), 891, replace=TRUE, prob=c(0.7,0.3))

# Em seguida associamos o valor mais predominante a amostra de treino e o outro, a amostra de teste.

train=titanic_train[amostra==0,]

test=titanic_train[amostra==1,]
```
## Criação dos modelos

Agora que selecionamos as variáveis, precisamos gerar um modelo.

Aqui, iremos usar técnicas de Random Forest, Naive Bayes, Regressão Logística e Support Vector Machine para a criação de vários modelos.
Todos esses modelos serão testados e aquele que exibir a melhor acurácia será utilizado para a predição final.

```{r}
# Primeiramente precisamos preparar os dados de teste.

dadospred = train %>% 
  mutate(PassengerId=NULL, Name=NULL, Ticket = NULL, Cabin=NULL) # Precisamos retirar as variáveis PassengerID, Name, Ticket e Cabin que não serão úteis para a predição.

modelorlg=glm(as.factor(dadospred$Survived) ~ ., family=binomial(link='logit'), dadospred)
modelorf=randomForest(as.factor(dadospred$Survived) ~ ., dadospred, ntree=100, importance=T)
modelosvm=svm(as.factor(dadospred$Survived) ~ ., dadospred)
modelonb = naiveBayes(as.factor(dadospred$Survived) ~ .,dadospred)
```
## Testando os modelos

Agora vamos fazer um teste com os dados da amostra de teste gerada anteriormente e checar a acurácia.

```{r message=FALSE}
test=test %>% 
  mutate(PassengerId=NULL, Name=NULL, Ticket = NULL, Cabin=NULL)

modelorlg$xlevels[["Prefix"]] <- union(modelorlg$xlevels[["Prefix"]], levels(test$Prefix))
modelorlg$xlevels[["Embarked"]] <- union(modelorlg$xlevels[["Embarked"]], levels(test$Embarked))

levels(test$Prefix)=levels(train$Prefix)
levels(test$Embarked)=levels(train$Embarked)

testrlg = predict(modelorlg, test, type="response")
testrf = predict(modelorf, test)
testsvm = predict(modelosvm, test)
testnb = predict(modelonb, test)

# Os resultados da regressão logísticas são mostrados na forma de probabilidade (0-1) e devem ser tranformados em 0 e 1.

a=0.50

testrlg[testrlg<a]=0
testrlg[testrlg>=a]=1

# Agora calculamos a acurácia dos modelos.

confusaorlg=table(testrlg,test$Survived)
confusaorf=table(testrf,test$Survived)
confusaosvm=table(testsvm,test$Survived)
confusaonb=table(testnb,test$Survived)

taxarlg=(confusaorlg[1]+confusaorlg[4])/sum(confusaorlg)
taxarf=(confusaorf[1]+confusaorf[4])/sum(confusaorf)
taxasvm=(confusaosvm[1]+confusaosvm[4])/sum(confusaosvm)
taxanb=(confusaonb[1]+confusaonb[4])/sum(confusaonb)

cat("Regressão Logística:", taxarlg,"\n")
cat("Random Forests:", taxarf,"\n")
cat("Support Vector Machine", taxasvm,"\n")
cat("Naive Bayes:", taxanb)
```

## Predição final

Agora passamos à predição final dos dados de teste.
Primeiramente, precisamos procurar por valores ausentes nos dados de teste.

```{r}
hNA = sapply(titanic_test, function(x) sum(is.na(x)))
names = names(hNA)
hNA = as.data.frame(hNA) %>% mutate(percent=hNA/nrow(titanic_train)*100)
row.names(hNA)=names
hNA[!hNA$percent==0,]
```

Como faltam muitos valores de Age, precisaremos usar o modelo criado anteriormente para prever esses valores.

```{r}
# Novamente preparamos os dados

dadospfinal = titanic_test %>% 
  mutate(Prefix = as.factor(str_sub(Name, regexpr(",", Name)+2,regexpr("\\.", Name)-1))) %>% 
  mutate(PassengerId=NULL, Name=NULL, Ticket = NULL, Cabin=NULL) 

levels(dadospfinal$Prefix)=levels(dadospred$Prefix)
levels(dadospfinal$Embarked)=levels(dadospred$Embarked)

# Agora fazemos a predição dos dados ausentes de teste a partir do modelo criado anteriormente.
    
  predrlteste = predict(modelorl, dadospfinal)
    
# Por fim substituímos os valores ausentes pelos valores preditos em Age.
    
    for (r in 1:nrow(dadospfinal)){if(is.na(dadospfinal[r,("Age")]==TRUE))
    {dadospfinal[r,("Age")]=predrlteste[r]}}

# E substituimos o valor ausente em Fare pela média.
  
  dadospfinal$Fare[is.na(dadospfinal$Fare)==TRUE]=mean(dadospfinal$Fare, na.rm = TRUE)
```

Em seguida, geramos as predições.

```{r message=FALSE, warning=FALSE}
resultrlg = predict(modelorlg, dadospfinal, type="response")
resultrf = predict(modelorf, dadospfinal)
resultsvm = predict(modelosvm, dadospfinal)
resultnb = predict(modelonb, dadospfinal)

a=0.50

resultrlg[resultrlg<a]=0
resultrlg[resultrlg>=a]=1
```

## Criando os arquivos .csv para submissão.

Agora, precisamos criar arquivos .csv com as predições para submissão.

```{r}
# Primeiro padronizamos as tabelas.

filerlg = data.frame(titanic_test$PassengerId,resultrlg)
names(filerlg)[1]="PassengerID"
names(filerlg)[2]="Survived"
filerf = data.frame(titanic_test$PassengerId,resultrf)
names(filerf)=names(filerlg)
filesvm = data.frame(titanic_test$PassengerId,resultsvm)
names(filesvm)=names(filerlg)
filenb = data.frame(titanic_test$PassengerId,resultnb)
names(filenb)=names(filerlg)

# Aqui os arquivos são salvos oferecendo a opção de escolha.

write.csv(filerlg, file.choose(), row.names=FALSE)
write.csv(filerf, file.choose(), row.names=FALSE)
write.csv(filesvm, file.choose(), row.names=FALSE)
write.csv(filenb, file.choose(), row.names=FALSE)
```

O maior valor de acurácia foi de 0.79847 pelo modelo que usou Support Vector Machine. Todos os outros modelos apresentaram resultados abaixo de 70% de acurácia.
