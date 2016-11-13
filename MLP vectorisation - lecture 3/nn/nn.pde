void setup(){
  
  // following codes were written only for testing purposes.
  LA la = new LA();
  MLP mlp = new MLP(3,2,15,2);
  
  double[][] A = { {1,1},{3,4},{5,0} };
  double[][] B = { {1,2},{4,5},{1,1} };
  
  Vector x = la.arrayToVector(A);
  Vector y = la.arrayToVector(B);
  
  //x.makeRandom(0,1);
  
  Vector D = mlp.feedForward(x);
  Vector BB = mlp.getCost(D,y);
  mlp.backPropogate(D,y);
  mlp.updateGradients();
  
  
  for(int i=0;i<3;i++){
    for(int j=0;j<2;j++){
      print(D.length()[0],D.length()[1],D.get(i,j)," ");
    }
    print("\n");
  }
  
  
  D = mlp.feedForward(x);
  BB = mlp.getCost(D,y);
  mlp.backPropogate(D,y);
  mlp.updateGradients();
  
  
  for(int i=0;i<3;i++){
    for(int j=0;j<2;j++){
      print(D.length()[0],D.length()[1],D.get(i,j)," ");
    }
    print("\n");
  }
  
  
}

void draw(){
  
}