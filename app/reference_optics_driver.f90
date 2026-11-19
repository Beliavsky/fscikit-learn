program reference_optics_driver
  use sklearn
  implicit none
  type(optics) :: model
  real(dp), allocatable :: x(:,:),d(:,:),train(:,:)
  integer, allocatable :: labels(:),hierarchy(:,:)
  integer :: n,p,i,j,input,output,k
  character(1024) :: path
  character(32) :: prefix,key
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),d(n,n))
  do i=1,n
    read(input,*) x(i,:)
  end do
  do i=1,n
    read(input,*) d(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,12
    model=optics(); model%min_samples=4; model%min_cluster_size=5
    model%metric='precomputed'; train=d
    select case(j)
    case(2); model%predecessor_correction=.false.
    case(3); model%xi=.15_dp
    case(4); model%max_eps=.7_dp
    case(5); model%min_samples_fraction=.12_dp; model%min_cluster_fraction=.18_dp
    case(6); model%min_cluster_size=0
    case(7); model%cluster_method='dbscan'; model%eps=.3_dp
    case(8); model%cluster_method='dbscan'; model%eps=.8_dp; model%max_eps=1
    case(9); model%metric='manhattan'; train=x
    case(10); model%metric='minkowski'; model%p=3; train=x
    case(11); model%max_eps=0
    case(12); model%min_samples=2; model%min_cluster_size=2
    end select
    call model%fit(train); write(prefix,'(a,i0)') 'optics',j
    call vector('_order',real(model%ordering_,dp))
    call vector('_predecessor',real(model%predecessor_,dp))
    call vector('_core',model%core_distances_)
    call vector('_reach',model%reachability_)
    call vector('_labels',real(model%labels_,dp))
    call vector('_hierarchy',real(reshape(model%cluster_hierarchy_,[size(model%cluster_hierarchy_)]),dp))
    call vector('_sizes',real([model%min_samples_,model%min_cluster_size_],dp))
    do k=1,3
      labels=cluster_optics_dbscan(model%reachability_,model%core_distances_,model%ordering_,real(k,dp)*.2_dp)
      write(key,'(a,i0)') '_dbscan',k
      call vector(trim(key),real(labels,dp))
    end do
    call cluster_optics_xi(model%reachability_,model%predecessor_,model%ordering_,model%min_samples_, &
      labels,hierarchy,xi=.1_dp,min_cluster_size=3,predecessor_correction=.false.)
    call vector('_rexi',real(labels,dp))
    call vector('_rehierarchy',real(reshape(hierarchy,[size(hierarchy)]),dp))
  end do
  close(output)
contains
  subroutine vector(suffix,v)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,v
  end subroutine
end program
