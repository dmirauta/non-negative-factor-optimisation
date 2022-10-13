# non-negative-factor-optimisation
A small base for non-negative matrix factorisation (NMF) and non-negative tensor factorisation (NTF) like models. Even for the same factor model and cost function, NMF methods can have many optimisation approaches, additional constraints, etc... The aim of this project was to serve as a basis for these variations. 

With this, factor models can be created by defining inidividual factors, their optimisation and how they are combined. The usual two factor models could then be specified for any cost by just providing the strictly positive and strictly negative terms of their gradients.
