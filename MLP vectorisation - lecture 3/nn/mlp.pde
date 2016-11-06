class MLP{
  int n_in;
  int n_hid;
  int n_out;
  int n_batch;
  
  Vector w1,w2,hiddenLayer;
  LA linearAlgebraObject = new LA();
  
  MLP(int batch, int in,int hid,int out){ // no bias node.. add at last time
    n_in = in;
    n_hid = hid;
    n_out = out;
    n_batch  = batch;
    
    w1 = new Vector(n_hid,n_in);
    w2 = new Vector(n_out,n_hid);
    hiddenLayer = new Vector(n_batch,n_hid);
    
    w1.makeRandom(0,1);
    w2.makeRandom(0,1);
  }
  
  Vector feedForward(Vector input){
    
    hiddenLayer = linearAlgebraObject.dot(input, linearAlgebraObject.trans(w1));
    hiddenLayer = linearAlgebraObject.sigmoid(hiddenLayer);
    return linearAlgebraObject.dot(hiddenLayer, linearAlgebraObject.trans(w2));
    
  }
  
  Vector getCost(Vector P, Vector y){
    
    Vector left = linearAlgebraObject.times(P,y);
    Vector right = linearAlgebraObject.times(  linearAlgebraObject.addScalar(y,1) , linearAlgebraObject.addScalar(  linearAlgebraObject.multScalar(P,-1)  ,1)  );
    return linearAlgebraObject.colSum(  linearAlgebraObject.rowSum( linearAlgebraObject.sub(left,right) )   );
  }
  
}