class MLP{
  int n_in;
  int n_hid;
  int n_out;
  int n_batch;
  
  Vector w1,w2,inputLayer,hiddenLayer,z1,z2,grad1,grad2;
  LA linearAlgebraObject = new LA();
  
  MLP(int batch, int in,int hid,int out){ // no bias node.. add at last time. make use of subSample() during backprop.
    n_in = in;
    n_hid = hid;
    n_out = out;
    n_batch  = batch;
    
    w1 = new Vector(n_hid,n_in);
    w2 = new Vector(n_out,n_hid);
    //hiddenLayer = new Vector(n_batch,n_hid);
    //z1 = new Vector(n_batch,n_hid);
    //z2 = new Vector(n_batch,n_out);
    
    w1.makeRandom(0,1);
    w2.makeRandom(0,1);
  }
  
  Vector feedForward(Vector input){
    inputLayer = input;
    z1 = linearAlgebraObject.dot(input, linearAlgebraObject.trans(w1));
    hiddenLayer = linearAlgebraObject.sigmoid(z1);
    z2 = linearAlgebraObject.dot(hiddenLayer, linearAlgebraObject.trans(w2));
    return linearAlgebraObject.sigmoid(z2);
  }
  
  Vector getCost(Vector P, Vector y){
    
    Vector left = linearAlgebraObject.times(P,y);
    Vector right = linearAlgebraObject.times(  linearAlgebraObject.addScalar(y,1) , linearAlgebraObject.addScalar(  linearAlgebraObject.multScalar(P,-1)  ,1)  );
    return linearAlgebraObject.colSum(  linearAlgebraObject.rowSum( linearAlgebraObject.sub(left,right) )   ); // divide by m
  }
  
  void backPropogate(Vector P,Vector y){
    Vector temp = linearAlgebraObject.sub(y,P);
    temp = linearAlgebraObject.times(temp, linearAlgebraObject.sigmoidGrad(z2));
    grad2 = linearAlgebraObject.dot(linearAlgebraObject.trans(temp),hiddenLayer); // scalar divide by m 
    
    temp = linearAlgebraObject.dot(temp, w2);
    // get rid of bias term from temp here, later
    temp = linearAlgebraObject.times(temp, linearAlgebraObject.sigmoidGrad(z1));
    grad1 = linearAlgebraObject.dot(linearAlgebraObject.trans(temp),inputLayer); // scalar divide by m 
  }
  
  void updateGradients(){
    grad1 = linearAlgebraObject.add(grad1,w1);
    grad2 = linearAlgebraObject.add(grad2,w2);
  }
  
}