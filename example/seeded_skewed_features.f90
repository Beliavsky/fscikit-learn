program seeded_skewed_features
  use sklearn,only: dp,skewed_chi2_sampler
  implicit none
  type(skewed_chi2_sampler)::model
  real(dp)::x(2,2)
  real(dp),allocatable::z(:,:)
  x=reshape([0._dp,1._dp,2._dp,3._dp],[2,2])
  model%n_components=4;model%random_state=42;model%skewedness=.5_dp
  z=model%fit_transform(x)
  print *, 'First weight row:',model%random_weights_(1,:)
  print *, 'Uniform offsets:',model%random_offset_
  print *, 'First feature row:',z(1,:)
  model%n_components=16;z=model%transform(x)
  print *, 'Rescaled feature row:',z(1,:)
end program
