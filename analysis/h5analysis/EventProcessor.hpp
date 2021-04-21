//
//  EventProcessor.hpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef EventProcessor_hpp
#define EventProcessor_hpp

#include "Helpers.hpp"
#include "Event.hpp"

class EventProcessor
{
public:
  EventProcessor(){}
  
  vector<string> getLabels(Group group);
  vector<shared_ptr<Event>> getValues(Group groupEvent, Group groupEFPs, Group groupJet, Group groupConstituents);
  
private:
  vector<shared_ptr<Event>> events;
  
  void fillEvents(Group groupEvent);
  void fillEFPs(Group groupEFPs);
  void fillJets(Group groupJet);
  void fillConstituents(Group group);
  
};




#endif /* EventProcessor_hpp */
