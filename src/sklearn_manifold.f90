! Isomap: sklearn 1.6.1 manifold/_isomap.py and utils/graph.py (BSD).
! Native dense/CSR precomputed neighborhoods and FW/Dijkstra geodesics.
module sklearn_manifold
  use sklearn_common
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_feature_extraction, only: feature_string
  use sklearn_ieee_math, only: ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_positive_inf
  use sklearn_tsne, only: tsne,trustworthiness
  use sklearn_metrics, only: pairwise_distances
  use sklearn_kernel_pca, only: kernel_pca
  use sklearn_spectral, only: spectral_embedding
  use sklearn_mds, only: mds
  use sklearn_smacof,only:smacof,smacof_result
  use sklearn_locally_linear, only: locally_linear_embedding
  implicit none
  private
  public :: smacof,smacof_result
  public :: isomap,spectral_embedding,mds,locally_linear_embedding,tsne,trustworthiness
  type :: isomap
    integer :: n_neighbors=5,n_components=2,n_connected_components_=0
    character(16) :: metric='minkowski'
    real(dp) :: p=2
    real(dp), allocatable :: x_fit_(:,:),dist_matrix_(:,:),embedding_(:,:)
    type(kernel_pca) :: kernel_pca_
    real(dp) :: radius=-1
    character(8) :: path_method='auto'
    integer :: n_features_in_=0,n_samples_fit_=0,n_features_out_=0
    character(8) :: path_method_='auto'
    integer, private :: neighbors_fit_=0
    real(dp), private :: radius_fit_=-1,p_fit_=2
    character(16), private :: metric_fit_='minkowski'
  contains
    procedure, private :: fit_dense=>isomap_fit,fit_sparse=>isomap_fit_sparse
    generic :: fit=>fit_dense,fit_sparse
    procedure, private :: fit_transform_dense=>isomap_fit_transform,fit_transform_sparse=>isomap_map_sparse
    generic :: fit_transform=>fit_transform_dense,fit_transform_sparse
    procedure, private :: transform_dense=>isomap_transform,transform_sparse=>isomap_transform_sparse
    generic :: transform=>transform_dense,transform_sparse
    procedure :: reconstruction_error=>isomap_error
    procedure :: get_feature_names_out=>isomap_names
  end type
contains
  subroutine parameters(self,n)
    class(isomap), intent(inout) :: self
    integer, intent(in) :: n
    call require(n>=1,'Isomap requires at least one sample')
    call require(.not.ieee_is_nan(self%radius),'Isomap radius is NaN')
    call require(ieee_is_finite(self%radius),'Isomap radius must be finite')
    call require(self%radius==-1.or.self%radius>=0,'invalid Isomap radius')
    call require(.not.ieee_is_nan(self%p),'Isomap p is NaN')
    call require(ieee_is_finite(self%p),'Isomap p must be finite')
    call require(self%p>=1,'Isomap p must be at least one')
    call require(self%n_neighbors>=0,'Isomap n_neighbors must be nonnegative')
    if(self%n_neighbors>0) then
      call require(self%radius<0,'Isomap requires either n_neighbors or radius, not both')
      call require(self%n_neighbors<n,'Isomap requires 0 < n_neighbors < n_samples')
    else
      call require(self%radius>=0,'Isomap radius must be nonnegative when n_neighbors=0')
    end if
    call require(self%n_components>0,'invalid Isomap n_components')
    call require(any(self%path_method==['auto    ','FW      ','D       ']),'invalid Isomap path_method')
    self%neighbors_fit_=self%n_neighbors;self%radius_fit_=self%radius
    self%metric_fit_=self%metric;self%p_fit_=self%p;self%n_samples_fit_=n
  end subroutine
  subroutine check_distances(x,n,training)
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: n
    logical, intent(in) :: training
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols==n,'Isomap precomputed distance shape mismatch')
    if(training) call require(x%n_rows==n,'Isomap precomputed fit distances must be square')
    call require(all(ieee_is_finite(x%data)),'Isomap distances must be finite')
    call require(all(x%data>=0),'Isomap distances must be nonnegative')
  end subroutine
  function selected_graph(self,input,training) result(graph)
    class(isomap), intent(in) :: self
    type(csr_matrix), intent(in) :: input
    logical, intent(in) :: training
    type(csr_matrix) :: graph
    integer, allocatable :: order(:),columns(:)
    real(dp), allocatable :: values(:)
    integer :: i,j,lo,hi,take,remove,slot,used,k
    call check_distances(input,self%n_samples_fit_,training)
    graph%n_rows=input%n_rows;graph%n_cols=input%n_cols
    allocate(graph%indptr(input%n_rows+1),columns(size(input%data)),values(size(input%data)))
    used=0;graph%indptr(1)=1
    do i=1,input%n_rows
      lo=input%indptr(i);hi=input%indptr(i+1)-1
      order=argsort(input%data(lo:hi));remove=0;take=size(order)
      if(self%neighbors_fit_>0) then
        take=self%neighbors_fit_;if(training) take=take+1
        call require(size(order)>=take,'not enough stored neighbors for Isomap')
        if(training) then
          do j=1,take
            if(input%indices(lo+order(j)-1)==i) then
              call require(remove==0,'duplicate selected self entries in Isomap neighbors');remove=j
            end if
          end do
          if(remove==0) remove=1
        end if
      end if
      do j=1,take
        slot=lo+order(j)-1;k=input%indices(slot)
        if(self%neighbors_fit_>0) then
          if(j==remove) cycle
        else
          if(training.and.k==i) cycle
          if(input%data(slot)>self%radius_fit_) cycle
        end if
        used=used+1;columns(used)=k;values(used)=input%data(slot)
      end do
      graph%indptr(i+1)=used+1
    end do
    graph%indices=columns(:used);graph%data=values(:used)
  end function
  subroutine isomap_fit(self,x)
    class(isomap), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:)
    type(csr_matrix) :: input,graph
    call check_x(x);call parameters(self,size(x,1))
    if(self%metric=='precomputed') then
      call require(size(x,1)==size(x,2),'Isomap precomputed fit distances must be square')
      d=x
    else
      d=pairwise_distances(x,x,self%metric_fit_,self%p_fit_)
    end if
    input=csr_from_dense(d,.true.);graph=selected_graph(self,input,.true.)
    self%x_fit_=x;self%n_features_in_=size(x,2)
    call embed(self,graph,d)
  end subroutine
  subroutine isomap_fit_sparse(self,x)
    class(isomap), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: graph
    call require(self%metric=='precomputed','CSR Isomap requires metric=precomputed')
    call x%validate();call require(x%n_rows==x%n_cols,'Isomap precomputed fit distances must be square')
    call parameters(self,x%n_rows);graph=selected_graph(self,x,.true.)
    if(allocated(self%x_fit_)) deallocate(self%x_fit_)
    self%n_features_in_=x%n_cols
    call embed(self,graph)
  end subroutine
  subroutine embed(self,neighbors,full_distances)
    class(isomap), intent(inout) :: self
    type(csr_matrix), intent(in) :: neighbors
    real(dp), intent(in), optional :: full_distances(:,:)
    real(dp), allocatable :: graph(:,:),g(:,:)
    integer, allocatable :: components(:),queue(:)
    real(dp) :: infinity,best
    integer :: n,i,j,t,c,head,tail,a,b,left,right
    n=neighbors%n_rows;infinity=ieee_value(0._dp,ieee_positive_inf)
    allocate(graph(n,n),components(n),queue(n));graph=infinity;components=0
    do i=1,n
      do t=neighbors%indptr(i),neighbors%indptr(i+1)-1
        j=neighbors%indices(t);graph(i,j)=min(graph(i,j),neighbors%data(t))
        graph(j,i)=min(graph(j,i),neighbors%data(t))
      end do
      graph(i,i)=0
    end do
    c=0
    do i=1,n
      if(components(i)/=0) cycle
      c=c+1;head=1;tail=1;queue(1)=i;components(i)=c
      do while(head<=tail)
        do j=1,n
          if(components(j)/=0) cycle
          if(.not.ieee_is_finite(graph(queue(head),j))) cycle
          tail=tail+1;queue(tail)=j;components(j)=c
        end do
        head=head+1
      end do
    end do
    self%n_connected_components_=c
    if(c>1) then
      call require(present(full_distances),'disconnected precomputed CSR Isomap graph cannot be completed')
      do a=2,c
        do b=1,a-1
          best=infinity;left=0;right=0
          do i=1,n
            if(components(i)/=a) cycle
            do j=1,n
              if(components(j)/=b) cycle
              if(full_distances(i,j)<best) then
                best=full_distances(i,j);left=i;right=j
              end if
            end do
          end do
          call require(left>0,'Isomap component distance is not finite')
          graph(left,right)=best;graph(right,left)=best
        end do
      end do
    end if
    self%path_method_=self%path_method
    if(self%path_method_=='auto') then
      self%path_method_='D'
      if(n<100.or.count(ieee_is_finite(graph))>=real(n,dp)*n/4) self%path_method_='FW'
    end if
    if(self%path_method_=='FW') then
      call floyd_warshall(graph);self%dist_matrix_=graph
    else
      self%dist_matrix_=dijkstra_all(graph)
    end if
    call require(all(ieee_is_finite(self%dist_matrix_)),'Isomap geodesics exceed finite working range')
    call require(maxval(self%dist_matrix_)<=sqrt(huge(1._dp))/4,'Isomap geodesics exceed kernel working range')
    self%kernel_pca_=kernel_pca();self%kernel_pca_%kernel='precomputed'
    self%kernel_pca_%n_components=self%n_components
    g=-0.5_dp*self%dist_matrix_**2;self%embedding_=self%kernel_pca_%fit_transform(g)
    self%n_features_out_=size(self%embedding_,2)
  end subroutine
  subroutine floyd_warshall(d)
    real(dp), intent(inout) :: d(:,:)
    integer :: i,j,k
    do k=1,size(d,1)
      do j=1,size(d,1)
        if(.not.ieee_is_finite(d(k,j))) cycle
        do i=1,size(d,1)
          if(.not.ieee_is_finite(d(i,k))) cycle
          d(i,j)=min(d(i,j),ieee_difference(d(i,k),-d(k,j)))
        end do
      end do
    end do
  end subroutine
  function dijkstra_all(graph) result(distances)
    real(dp), intent(in) :: graph(:,:)
    real(dp), allocatable :: distances(:,:),keys(:),values(:)
    integer, allocatable :: pointers(:),columns(:),heap(:),position(:)
    integer :: n,i,j,t,used,source,active,at,parent,child,node,neighbor,swap
    real(dp) :: candidate
    n=size(graph,1);allocate(distances(n,n),keys(n),pointers(n+1),heap(n),position(n))
    allocate(columns(count(ieee_is_finite(graph))),values(count(ieee_is_finite(graph))))
    used=0;pointers(1)=1
    do i=1,n
      do j=1,n
        if(.not.ieee_is_finite(graph(i,j)).or.i==j) cycle
        used=used+1;columns(used)=j;values(used)=graph(i,j)
      end do
      pointers(i+1)=used+1
    end do
    do source=1,n
      keys=ieee_value(0._dp,ieee_positive_inf);position=0;keys(source)=0
      active=1;heap(1)=source;position(source)=1
      do while(active>0)
        node=heap(1);position(node)=-1
        heap(1)=heap(active);active=active-1
        if(active>0) then
          position(heap(1))=1;at=1
          do while(2*at<=active)
            child=2*at
            if(child<active) then
              if(keys(heap(child+1))<keys(heap(child))) child=child+1
            end if
            if(keys(heap(at))<=keys(heap(child))) exit
            swap=heap(at);heap(at)=heap(child);heap(child)=swap
            position(heap(at))=at;position(heap(child))=child;at=child
          end do
        end if
        do t=pointers(node),pointers(node+1)-1
          neighbor=columns(t)
          if(position(neighbor)<0) cycle
          candidate=ieee_difference(keys(node),-values(t))
          if(candidate>=keys(neighbor)) cycle
          keys(neighbor)=candidate;at=position(neighbor)
          if(at==0) then
            active=active+1;at=active;heap(at)=neighbor;position(neighbor)=at
          end if
          do while(at>1)
            parent=at/2
            if(keys(heap(parent))<=keys(heap(at))) exit
            swap=heap(parent);heap(parent)=heap(at);heap(at)=swap
            position(heap(parent))=parent;position(heap(at))=at;at=parent
          end do
        end do
      end do
      distances(source,:)=keys
    end do
  end function
  function isomap_fit_transform(self,x) result(z)
    class(isomap), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x);z=self%embedding_
  end function
  function isomap_map_sparse(self,x) result(z)
    class(isomap), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: z(:,:)
    call self%fit(x);z=self%embedding_
  end function
  function isomap_transform(self,x) result(z)
    class(isomap), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),d(:,:)
    type(csr_matrix) :: input
    call require(allocated(self%embedding_),'Isomap is not fitted');call check_x(x)
    call require(size(x,2)==self%n_features_in_,'Isomap query feature count mismatch')
    if(self%metric_fit_=='precomputed') then
      d=x
    else
      d=pairwise_distances(x,self%x_fit_,self%metric_fit_,self%p_fit_)
    end if
    input=csr_from_dense(d,.true.);z=project(self,input)
  end function
  function isomap_transform_sparse(self,x) result(z)
    class(isomap), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%embedding_),'Isomap is not fitted')
    call require(self%metric_fit_=='precomputed','CSR Isomap query requires metric=precomputed')
    z=project(self,x)
  end function
  function project(self,input) result(z)
    class(isomap), intent(in) :: self
    type(csr_matrix), intent(in) :: input
    real(dp), allocatable :: z(:,:),geodesic(:,:)
    type(csr_matrix) :: neighbors
    real(dp) :: candidate
    integer :: i,j,t,k
    neighbors=selected_graph(self,input,.false.)
    allocate(geodesic(input%n_rows,self%n_samples_fit_));geodesic=ieee_value(0._dp,ieee_positive_inf)
    do i=1,input%n_rows
      call require(neighbors%indptr(i+1)>neighbors%indptr(i),'Isomap query has no neighbors within radius')
      do t=neighbors%indptr(i),neighbors%indptr(i+1)-1
        k=neighbors%indices(t)
        do j=1,self%n_samples_fit_
          candidate=ieee_difference(neighbors%data(t),-self%dist_matrix_(k,j))
          geodesic(i,j)=min(geodesic(i,j),candidate)
        end do
      end do
    end do
    call require(all(ieee_is_finite(geodesic)),'Isomap query geodesics exceed finite working range')
    call require(maxval(geodesic)<=sqrt(huge(1._dp))/4,'Isomap query geodesics exceed kernel working range')
    z=self%kernel_pca_%transform(-0.5_dp*geodesic**2)
  end function
  function isomap_names(self,input_features) result(names)
    class(isomap), intent(in) :: self
    character(*), intent(in), optional :: input_features(:)
    type(feature_string), allocatable :: names(:)
    character(32) :: suffix
    integer :: i
    call require(allocated(self%embedding_),'Isomap is not fitted')
    if(present(input_features)) call require(size(input_features)==self%n_features_in_,'Isomap input feature name count mismatch')
    allocate(names(self%n_features_out_))
    do i=1,size(names)
      write(suffix,'(i0)') i-1;names(i)%text='isomap'//trim(suffix)
    end do
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
