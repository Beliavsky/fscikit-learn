! Native composition re-exports corresponding to sklearn.pipeline.
module sklearn_pipeline
  use sklearn_feature_composition, only: feature_union,transformer_branch
  implicit none
  private
  public :: feature_union,transformer_branch
end module
