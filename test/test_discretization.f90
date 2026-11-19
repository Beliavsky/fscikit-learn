program test_discretization
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(kbins_discretizer) :: bins
  real(dp) :: x(9,2),q(5,2),w(9)
  real(dp), allocatable :: z(:,:),back(:,:),again(:,:)
  integer :: i
  x(:,1)=[(real(i-1,dp),i=1,9)]; x(:,2)=x(:,1)+10
  bins%n_bins=4; bins%strategy='uniform'; bins%encode='ordinal'
  z=bins%fit_transform(x)
  call require(all(bins%n_bins_==4),'uniform bin counts')
  call require(maxval(abs(bins%bin_edges_(1)%values-[0._dp,2._dp,4._dp,6._dp,8._dp]))==0, &
    'uniform bin edges')
  call require(all(z(:,1)==[0._dp,0._dp,1._dp,1._dp,2._dp,2._dp,3._dp,3._dp,3._dp]), &
    'right-sided ordinal bins')
  q(:,1)=[-100._dp,2._dp,4._dp,6._dp,100._dp]; q(:,2)=q(:,1)+10
  z=bins%transform(q)
  call require(all(z(:,1)==[0._dp,1._dp,2._dp,3._dp,3._dp]),'boundary and out-of-range bins')
  back=bins%inverse_transform(z); again=bins%transform(back)
  call require(all(z==again),'ordinal inverse preserves bins')
  bins%encode='onehot-dense'; z=bins%fit_transform(x,n_bins=[3,4])
  call require(size(z,2)==7.and.bins%n_features_out_==7,'heterogeneous dense output width')
  call require(all(sum(z,dim=2)==2),'one active bin per feature')
  back=bins%inverse_transform(z); again=bins%transform(back)
  call require(all(z==again),'one-hot inverse preserves bins')
  bins%encode='ordinal'; bins%strategy='quantile'; w=1; w(1)=0; w(9)=0
  call bins%fit(x,sample_weight=w)
  call require(all(bins%bin_edges_(1)%values==[1._dp,2._dp,4._dp,6._dp,7._dp]), &
    'weighted inverse-CDF quantiles ignore zero-weight endpoints')
  x(:,1)=[0._dp,0._dp,0._dp,0._dp,1._dp,1._dp,2._dp,2._dp,2._dp]
  call bins%fit(x)
  call require(bins%n_bins_(1)==2,'repeated quantiles prune zero-width bins')
  call require(all(bins%bin_edges_(1)%values==[0._dp,1._dp,2._dp]),'pruned quantile edges')
  x(:,1)=[0._dp,0.1_dp,0.2_dp,5._dp,5.1_dp,5.2_dp,10._dp,10.1_dp,10.2_dp]
  x(:,2)=3; bins%strategy='kmeans'; bins%n_bins=3
  z=bins%fit_transform(x); back=bins%inverse_transform(z)
  call require(bins%converged_,'one-dimensional Lloyd convergence')
  call require(maxval(abs(bins%bin_edges_(1)%values-[0._dp,2.6_dp,7.6_dp,10.2_dp]))<1.e-12_dp, &
    'kmeans midpoint edges')
  call require(bins%n_bins_(2)==1.and.all(z(:,2)==0),'constant feature is a single zero bin')
  call require(all(ieee_is_nan(back(:,2))),'constant inverse follows infinite-edge NaN convention')
  w=1; w(1)=2; w(9)=0; call bins%fit(x,sample_weight=w)
  call require(abs(bins%bin_edges_(1)%values(2)-2.5875_dp)<1.e-12_dp,'weighted kmeans centers')
  ! Duplicate observations exercise empty cluster relocation and pruning.
  x(:,1)=[0._dp,0._dp,0._dp,0._dp,0._dp,10._dp,10._dp,10._dp,10._dp]
  call bins%fit(x)
  call require(bins%converged_.and.bins%n_bins_(1)>=1,'duplicate kmeans observations')
  z=bins%transform(x); call require(all(z>=0),'duplicate bins are valid')
  bins%strategy='uniform'; bins%encode='onehot-dense'
  again=bins%transform(x); call require(all(again==z),'fitted encoding retained until refit')
  print *, 'KBinsDiscretizer tests passed'
end program
