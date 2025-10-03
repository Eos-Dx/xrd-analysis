#ifndef RunAction_h
#define RunAction_h 1

#include "G4UserRunAction.hh"
#include "G4Accumulable.hh"
#include "globals.hh"

class G4Run;

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    virtual ~RunAction();

    virtual G4Run* GenerateRun() override;
    virtual void BeginOfRunAction(const G4Run*) override;
    virtual void EndOfRunAction(const G4Run*) override;

    void AddEdep (G4double edep);

private:
    G4Accumulable<G4double> fEdep;
    G4Accumulable<G4double> fEdep2;
};

#endif
