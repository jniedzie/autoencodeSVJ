//
//  Constituent.cpp
//  h5analysis
//
//  Created by Jeremi Niedziela on 21/04/2021.
//  Copyright © 2021 Jeremi Niedziela. All rights reserved.
//

#include "Constituent.hpp"


void Constituent::print()
{
  cout<<"Constituent:"<<endl;
  cout<<"\teta: "<<eta<<"\tphi: "<<phi<<"\tpt: "<<pt<<"\tenergy: "<<energy<<"\trapidity: "<<rapidity<<endl;;
  
  cout<<endl;
}

bool Constituent::isEmpty()
{
  return eta == 0 && phi == 0 && pt == 0;
}
