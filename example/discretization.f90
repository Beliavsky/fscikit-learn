program discretization
  use sklearn
  implicit none
  type(kbins_discretizer) :: bins
  real(dp) :: x(8,1),query(3,1)
  real(dp), allocatable :: codes(:,:),centers(:,:)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,6._dp,8._dp,12._dp]
  query(:,1)=[-1._dp,3._dp,15._dp]
  bins%n_bins=3; bins%encode='ordinal'; bins%strategy='quantile'
  call bins%fit(x)
  codes=bins%transform(query); centers=bins%inverse_transform(codes)
  print *, 'Learned quantile edges:',bins%bin_edges_(1)%values
  print *, 'Zero-based bins:',codes(:,1)
  print *, 'Approximate inverse (bin centers):',centers(:,1)
end program
