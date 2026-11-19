program reference_birch_driver
  use sklearn
  implicit none
  type(birch) :: model
  real(dp), allocatable :: x(:,:),query(:,:),z(:,:)
  integer, allocatable :: labels(:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input); query=x+.137_dp
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,10
    model=birch(); model%threshold=.18_dp; model%branching_factor=3; model%n_clusters=0
    select case(j)
    case(2); model%threshold=.04_dp; model%branching_factor=2
    case(3); model%branching_factor=50
    case(4); model%n_clusters=3
    case(5); model%threshold=100; model%n_clusters=8
    case(6); model%compute_labels=.false.
    case(7); model%threshold=.04_dp; model%branching_factor=2
    case(8); model%n_clusters=4
    case(9); model%threshold=.4_dp; model%branching_factor=2
    case(10); model%threshold=.001_dp; model%branching_factor=2
    end select
    if(j==7.or.j==8) then
      do i=1,n,7
        call model%partial_fit(x(i:min(n,i+6),:))
      end do
      if(j==8) then
        model%n_clusters=2; call model%partial_fit()
      end if
    else
      call model%fit(x)
    end if
    write(prefix,'(a,i0)') 'birch',j
    z=model%subcluster_centers_; call vector('_centers',reshape(z,[size(z)]))
    call vector('_sub_labels',real(model%subcluster_labels_,dp))
    if(allocated(model%labels_)) call vector('_labels',real(model%labels_,dp))
    labels=model%predict(x); call vector('_train',real(labels,dp))
    labels=model%predict(query); call vector('_query',real(labels,dp))
    z=model%transform(x); call vector('_distances',reshape(z,[size(z)]))
    z=model%transform(query); call vector('_query_distances',reshape(z,[size(z)]))
    call vector('_metadata',real([model%n_samples_seen_,model%n_features_in_, &
      merge(1,0,model%insufficient_subclusters_)],dp))
  end do
  close(output)
contains
  subroutine vector(suffix,v)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,v
  end subroutine
end program
