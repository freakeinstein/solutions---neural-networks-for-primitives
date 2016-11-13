void setup(){
  
  // following codes were written only for testing purposes.
  LA la = new LA();
  MLP mlp = new MLP(3,2,15,2);
  
  double[][] A = { {1,1},{3,4},{5,0} }; // to do: make good examples to test training.
  double[][] B = { {1,2},{4,5},{1,1} };
  
  Vector x = la.arrayToVector(A);
  Vector y = la.arrayToVector(B);
  
  //x.makeRandom(0,1);
  
  mlp.clearGradientsToZero();
  
  int counter = 0;
  while(true){ // running epoch.
  
    Vector D = mlp.feedForward(x); // predict.    
    mlp.backPropogate(D,y); // during backprop. gradients will only get accumulated not updated.
    
    
    if(counter % 10 == 0){ 
      Vector cost = mlp.getCost(D,y); // calculate cost.
      print("\n cost: ",cost.get(0,0)," ");
      
      mlp.updateGradients(); // update weights based on previous online learning.
      mlp.clearGradientsToZero(); // clear gradient accumulation after weights updation. IMPORTANT.
      
      //showVectorContents(D);
    
      counter = 0;
    }
    
  }
  
  
}

void showVectorContents(Vector v){
  int r = v.length()[0];
  int c = v.length()[1];
  
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      print(v.get(i,j)," ");
    }
    print("\n");
  }
  print("\n");
}

void draw(){
  
}