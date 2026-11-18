! sklearn 1.6.1 manifold/_spectral_embedding.py; see THIRD_PARTY_NOTICES.md.
module sklearn_spectral
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances,rbf_kernel
  implicit none
  private
  public :: spectral_embedding,spectral_affinity_matrix,spectral_eigenvectors
  type :: spectral_embedding
    integer :: n_components=2,n_neighbors=0,n_neighbors_=0,n_connected_components_=0
    real(dp) :: gamma=-1,gamma_=0
    character(20) :: affinity='nearest_neighbors'
    real(dp), allocatable :: affinity_matrix_(:,:),embedding_(:,:)
  contains
    procedure :: fit=>embedding_fit
    procedure :: fit_transform=>embedding_fit_transform
  end type
contains
  function spectral_affinity_matrix(x,affinity,gamma,n_neighbors) result(a)
    real(dp), intent(in) :: x(:,:),gamma
    character(*), intent(in) :: affinity
    integer, intent(in) :: n_neighbors
    real(dp), allocatable :: a(:,:),d(:,:)
    integer :: n,i,order(size(x,1))
    call check_x(x); n=size(x,1)
    select case(affinity)
    case('rbf')
      call check_x(reshape([gamma],[1,1]))
      call require(gamma>0,'spectral gamma must be positive'); a=rbf_kernel(x,x,gamma)
    case('nearest_neighbors')
      call require(n_neighbors>=1.and.n_neighbors<=n,'invalid spectral n_neighbors')
      d=pairwise_distances(x,x,'sqeuclidean'); allocate(a(n,n)); a=0
      do i=1,n
        order=argsort(d(i,:)); a(i,order(:n_neighbors))=1
      end do
      a=0.5_dp*(a+transpose(a))
    case('precomputed')
      call require(size(x,2)==n,'spectral affinity must be square'); a=x
    case default
      call require(.false.,'unknown spectral affinity')
    end select
    call require(all(a>=0),'spectral affinity must be nonnegative')
    call require(all(abs(a-transpose(a))<=1.e-10_dp*max(1._dp,abs(a),abs(transpose(a)))), &
      'spectral affinity must be symmetric')
    a=0.5_dp*(a+transpose(a))
  end function
  function spectral_eigenvectors(a,n_components,drop_first,n_connected) result(z)
    real(dp), intent(in) :: a(:,:)
    integer, intent(in) :: n_components
    logical, intent(in) :: drop_first
    integer, intent(out) :: n_connected
    real(dp), allocatable :: z(:,:)
    real(dp) :: lap(size(a,1),size(a,1)),degree(size(a,1)),values(size(a,1)),v(size(a,1),size(a,1))
    logical :: visited(size(a,1))
    integer :: n,i,j,first,queue(size(a,1)),head,tail,k
    n=size(a,1); first=1
    if(drop_first) first=2
    call require(n_components>0.and.n_components+first-1<=n,'invalid spectral n_components')
    lap=-a
    do i=1,n
      lap(i,i)=0
    end do
    ! Exclude self loops before summation: subtracting a large diagonal after
    ! a row sum can erase the off-diagonal degree through cancellation.
    degree=max(0._dp,-sum(lap,dim=2))
    degree=sqrt(degree); where(degree==0) degree=1
    lap=lap/(spread(degree,2,n)*spread(degree,1,n))
    do i=1,n
      lap(i,i)=-sum(lap(i,:)*degree)/degree(i)
    end do
    ! The diagonal above is one for nodes with edges, zero for isolated nodes.
    call symmetric_eigen(-lap,values,v)
    z=v(:,first:first+n_components-1)/spread(degree,2,n_components)
    do j=1,n_components
      i=maxloc(abs(z(:,j)),dim=1)
      if(z(i,j)<0) z(:,j)=-z(:,j)
    end do
    visited=.false.; n_connected=0
    do i=1,n
      if(visited(i)) cycle
      n_connected=n_connected+1; head=1; tail=1; queue(1)=i; visited(i)=.true.
      do while(head<=tail)
        k=queue(head); head=head+1
        do j=1,n
          if(visited(j).or.a(k,j)<=0) cycle
          tail=tail+1; queue(tail)=j; visited(j)=.true.
        end do
      end do
    end do
  end function
  subroutine embedding_fit(self,x)
    class(spectral_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x); self%gamma_=self%gamma
    if(self%gamma==-1) self%gamma_=1._dp/size(x,2)
    self%n_neighbors_=self%n_neighbors
    if(self%n_neighbors==0) self%n_neighbors_=max(1,size(x,1)/10)
    self%affinity_matrix_=spectral_affinity_matrix(x,self%affinity,self%gamma_,self%n_neighbors_)
    self%embedding_=spectral_eigenvectors(self%affinity_matrix_,self%n_components,.true.,self%n_connected_components_)
  end subroutine
  function embedding_fit_transform(self,x) result(z)
    class(spectral_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%embedding_
  end function
end module
