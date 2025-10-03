#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

class RunAction;

class EventAction : public G4UserEventAction
{
public:
    EventAction(RunAction* runAction);
    virtual ~EventAction();

    virtual void BeginOfEventAction(const G4Event* event) override;
    virtual void EndOfEventAction(const G4Event* event) override;

    void AddEdep(G4double edep) { fEdep += edep; }

private:
    RunAction* fRunAction;
    G4double   fEdep;
};

#endif
