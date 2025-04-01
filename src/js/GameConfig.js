/**
 * Configuration class for game generation parameters
 */
class GameConfig {
  constructor({
    genMode = 'init',
    parents = [],
    saveDir = '',
    expSeed = 0,
    fewshot = true,
    cot = true,
    fromIdea = false,
    idea = '',
    fromPlan = false,
    maxGenAttempts = 10,
    metaParents = null
  } = {}) {
    this.genMode = genMode;
    this.parents = parents;
    this.saveDir = saveDir;
    this.expSeed = expSeed;
    this.fewshot = fewshot;
    this.cot = cot;
    this.fromIdea = fromIdea;
    this.idea = idea;
    this.fromPlan = fromPlan;
    this.maxGenAttempts = maxGenAttempts;
    this.metaParents = metaParents;
  }

  // Create a new config by extending this one with new values
  extend(newValues) {
    return new GameConfig({
      ...this,
      ...newValues
    });
  }

  // Create a standard config with experiment seed and save directory
  static forExperiment(expSeed, saveDir) {
    return new GameConfig({
      expSeed,
      saveDir
    });
  }
}
