! Isomap with a dense neighbor graph and Floyd-Warshall shortest paths.
module sklearn_manifold
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances
  use sklearn_kernel_pca, only: kernel_pca
  use sklearn_spectral, only: spectral_embedding
  use sklearn_mds, only: mds
  use sklearn_locally_linear, only: locally_linear_embedding
  implicit none
  private
  public :: isomap,spectral_embedding,mds,locally_linear_embedding
  type :: isomap
    integer :: n_neighbors=5,n_components=2,n_connected_components_=0
    character(16) :: metric='minkowski'
    real(dp) :: p=2
    real(dp), allocatable :: x_fit_(:,:),dist_matrix_(:,:),embedding_(:,:)
    type(kernel_pca) :: kernel_pca_
  contains
    procedure :: fit=>isomap_fit
    procedure :: fit_transform=>isomap_fit_transform
    procedure :: transform=>isomap_transform
    procedure :: reconstruction_error=>isomap_error
  end type
contains
  subroutine shortest_paths(d)
    real(dp), intent(inout) :: d(:,:)
    integer :: i,j,k
    do k=1,size(d,1)
      do j=1,size(d,1)
        do i=1,size(d,1)
          d(i,j)=min(d(i,j),d(i,k)+d(k,j))
        end do
      end do
    end do
  end subroutine
  subroutine isomap_fit(self,x)
    class(isomap), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:),graph(:,:),row(:),g(:,:)
    integer, allocatable :: order(:),components(:)
    integer :: n,i,j,k,c,a,b,left,right
    real(dp) :: infinity,best
    call check_x(x); n=size(x,1); k=self%n_neighbors
    call require(k>0.and.k<n,'Isomap requires 0 < n_neighbors < n_samples')
    call require(self%n_components>0.and.self%n_components<n,'invalid Isomap n_components')
    self%x_fit_=x; d=pairwise_distances(x,x,self%metric,self%p)
    infinity=huge(1._dp)/4; allocate(graph(n,n),components(n)); graph=infinity; components=0
    do i=1,n
      row=d(i,:); row(i)=infinity; order=argsort(row)
      graph(i,order(:k))=d(i,order(:k)); graph(order(:k),i)=d(i,order(:k)); graph(i,i)=0
    end do
    call shortest_paths(graph)
    c=0
    do i=1,n
      if(components(i)/=0) cycle
      c=c+1
      where(graph(i,:)<infinity) components=c
    end do
    self%n_connected_components_=c
    ! Match sklearn's completion: connect the closest pair for every pair of
    ! original components, then recompute all-pairs shortest paths.
    if(c>1) then
      do a=1,c-1
        do b=a+1,c
          best=infinity; left=0; right=0
          do i=1,n
            if(components(i)/=a) cycle
            do j=1,n
              if(components(j)/=b) cycle
              if(d(i,j)<best) then
                best=d(i,j); left=i; right=j
              end if
            end do
          end do
          graph(left,right)=best; graph(right,left)=best
        end do
      end do
      call shortest_paths(graph)
    end if
    self%dist_matrix_=graph; self%kernel_pca_%kernel='precomputed'
    self%kernel_pca_%n_components=self%n_components
    g=-0.5_dp*graph**2; self%embedding_=self%kernel_pca_%fit_transform(g)
  end subroutine
  function isomap_fit_transform(self,x) result(z)
    class(isomap), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%embedding_
  end function
  function isomap_transform(self,x) result(z)
    class(isomap), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),distances(:,:),geodesic(:,:),row(:)
    integer, allocatable :: order(:)
    integer :: i,j,n,k
    call require(allocated(self%x_fit_),'Isomap is not fitted')
    distances=pairwise_distances(x,self%x_fit_,self%metric,self%p)
    n=size(self%x_fit_,1); k=self%n_neighbors; allocate(geodesic(size(x,1),n))
    do i=1,size(x,1)
      order=argsort(distances(i,:))
      do j=1,n
        geodesic(i,j)=minval(distances(i,order(:k))+self%dist_matrix_(order(:k),j))
      end do
    end do
    z=self%kernel_pca_%transform(-0.5_dp*geodesic**2)
  end function
  real(dp) function isomap_error(self) result(error)
    class(isomap), intent(in) :: self
    real(dp), allocatable :: g(:,:),means(:)
    integer :: n
    call require(allocated(self%dist_matrix_),'Isomap is not fitted')
    n=size(self%dist_matrix_,1); g=-0.5_dp*self%dist_matrix_**2; means=sum(g,dim=1)/n
    g=g-spread(means,1,n)-spread(means,2,n)+sum(means)/n
    error=sqrt(max(0._dp,sum(g*g)-sum(self%kernel_pca_%eigenvalues_**2)))/n
  end function
end module
