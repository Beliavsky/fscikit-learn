program test_birch
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(birch) :: batch,stream,copy
  real(dp) :: x(120,2),one(1,2)
  real(dp), allocatable :: d(:,:),centers(:,:)
  integer, allocatable :: labels(:)
  integer :: i,j
  do i=1,120
    x(i,:)=[sin(real(i,dp))*3,cos(real(i*7,dp))*2]+real(mod(i,3),dp)*6
  end do
  batch%threshold=.08_dp; batch%branching_factor=2; batch%n_clusters=0
  call batch%fit(x)
  call require(size(batch%subcluster_centers_,1)>50,'small branching forces deep CF tree')
  call require(batch%n_samples_seen_==120,'sample counter')
  stream%threshold=batch%threshold; stream%branching_factor=2; stream%n_clusters=0
  do i=1,120,7
    call stream%partial_fit(x(i:min(i+6,120),:))
  end do
  call require(all(shape(stream%subcluster_centers_)==shape(batch%subcluster_centers_)),'stream subcluster count')
  call require(maxval(abs(stream%subcluster_centers_-batch%subcluster_centers_))<1.e-13_dp,'stream equals batch')
  call require(size(stream%labels_)==1,'partial labels describe latest batch')
  labels=stream%predict(x)
  call require(all(labels==batch%labels_),'stream predictions equal batch')
  copy=stream; centers=copy%subcluster_centers_
  stream%n_clusters=3; call stream%partial_fit()
  call require(size(stream%labels_)==1,'global-only update retains last batch labels')
  labels=stream%predict(x)
  call require(size(unique_values(labels))==3,'Ward groups subclusters')
  call require(all(copy%subcluster_centers_==centers),'model copies own tree storage')
  d=batch%transform(x)
  do i=1,120
    j=minloc(d(i,:),dim=1)
    call require(batch%labels_(i)==j-1,'nearest subcluster predicts labels')
  end do
  batch%threshold=100; batch%n_clusters=3; call batch%fit(x)
  call require(size(batch%subcluster_centers_,1)==1.and.batch%insufficient_subclusters_,'too few centroids fallback')
  call require(maxval(abs(batch%subcluster_centers_(1,:)-sum(x,dim=1)/120))<1.e-13_dp,'CF centroid statistics')
  call require(batch%n_samples_seen_==120,'fit resets tree')
  one=2; batch%compute_labels=.false.; call batch%fit(one)
  call require(.not.allocated(batch%labels_),'compute_labels false skips labels')
  labels=batch%predict(one); call require(all(labels==0),'single point prediction')
  d=batch%fit_transform(one); call require(all(d==0),'fit_transform single point')
  print *, 'BIRCH tests passed'
contains
  function unique_values(values) result(u)
    use sklearn_common, only: unique_labels
    integer, intent(in) :: values(:)
    integer, allocatable :: u(:)
    u=unique_labels(values)
  end function
end program
