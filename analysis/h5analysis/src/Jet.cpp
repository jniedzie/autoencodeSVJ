//
//  Jet.cpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "Jet.hpp"

void Jet::print()
{
  cout<<"Jet:"<<endl;
  cout<<"\teta: "<<eta<<"\tphi: "<<phi<<"\tpt: "<<pt<<"\tmass: "<<mass<<"\tcharged fraction: "<<chargedFraction;
  cout<<"\tPTD: "<<PTD<<"\taxis minor: "<<axisMinor<<"\tflavor: "<<flavor<<"\tenergy: "<<energy<<endl;
  cout<<"\tEFPs:";
  for(double EFP : EFPs) cout<<EFP<<"\t";
  
  for(auto constituent : constituents) constituent->print();
  
  cout<<endl;
}

bool Jet::isEmpty()
{
  return eta==0 && phi==0 && pt==0;
}
