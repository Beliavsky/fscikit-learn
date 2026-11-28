! sklearn 1.6.1 manifold/_spectral_embedding.py; see THIRD_PARTY_NOTICES.md.
module sklearn_spectral
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances,rbf_kernel
  use sklearn_sparse,only:csr_matrix
  use sklearn_neighbor_functions,only:kneighbors_graph
  use sklearn_kernel_dispatch,only:pairwise_kernels
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_ieee_math,only:ieee_product,ieee_total
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public :: spectral_embedding,spectral_affinity_matrix,spectral_eigenvectors
  public :: spectral_discretize
  type :: spectral_embedding
    integer :: n_components=2,n_neighbors=0,n_neighbors_=0,n_connected_components_=0
    real(dp) :: gamma=-1,gamma_=0
    character(32) :: affinity='nearest_neighbors'
    real(dp), allocatable :: affinity_matrix_(:,:),embedding_(:,:)
  contains
    procedure :: fit_dense=>embedding_fit,fit_csr=>embedding_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure :: fit_transform_dense=>embedding_fit_transform,fit_transform_csr=>embedding_fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
  end type
  interface spectral_affinity_matrix
    module procedure affinity_dense,affinity_csr
  end interface
contains
  function spectral_discretize(vectors,random_state,n_iter_max,max_svd_restarts,rng)result(labels)
    ! sklearn.cluster._spectral.discretize (BSD-3-Clause).
    real(dp),intent(in)::vectors(:,:)
    integer,optional,intent(in)::random_state,n_iter_max,max_svd_restarts
    type(legacy_random_state),optional,intent(inout)::rng
    integer::labels(size(vectors,1))
    type(legacy_random_state)::state
    real(dp),allocatable::work(:,:),rotation(:,:),projected(:,:),cross(:,:),s(:),u(:,:),v(:,:),c(:)
    real(dp),allocatable::candidate(:),best(:)
    real(dp)::norm,objective,last_objective,best_norm
    integer::n,k,i,j,l,iteration,limit,restarts,max_restarts,seed
    logical::converged
    call check_x(vectors);n=size(vectors,1);k=size(vectors,2)
    call require(.not.(present(rng).and.present(random_state)),'discretize accepts a seed or RNG state, not both')
    if(present(rng))then
      state=rng
    else
      seed=0;if(present(random_state))seed=random_state
      call state%seed(seed)
    end if
    limit=20;if(present(n_iter_max))limit=n_iter_max
    max_restarts=30;if(present(max_svd_restarts))max_restarts=max_svd_restarts
    call require(max_restarts>0,'discretize SVD did not converge')
    work=vectors
    do j=1,k
      norm=sqrt(ieee_total([(ieee_product(work(i,j),work(i,j)),i=1,n)]))
      call require(norm>0,'discretize requires nonzero column norms')
      work(:,j)=(work(:,j)/norm)*sqrt(real(n,dp))
      if(work(1,j)/=0)work(:,j)=-work(:,j)*sign(1._dp,work(1,j))
    end do
    do i=1,n
      norm=sqrt(sum(work(i,:)**2))
      call require(norm>0,'discretize requires nonzero row norms')
      work(i,:)=work(i,:)/norm
    end do
    allocate(rotation(k,k),cross(k,k),s(k),u(k,k),v(k,k),c(n),candidate(k),best(k))
    restarts=0
    do while(restarts<max_restarts)
      rotation=0;rotation(:,1)=work(state%integer_index(n),:);c=0
      do j=2,k
        c=c+abs(matmul(work,rotation(:,j-1)))
        rotation(:,j)=work(minloc(c,dim=1),:)
      end do
      last_objective=0;iteration=0
      do
        iteration=iteration+1;projected=matmul(work,rotation)
        labels=maxloc(projected,dim=2)-1
        cross=0
        do i=1,n
          cross(labels(i)+1,:)=cross(labels(i)+1,:)+work(i,:)
        end do
        call dense_svd_status(cross,s,u,v,converged)
        if(.not.converged)then
          restarts=restarts+1;exit
        end if
        objective=2*(n-sum(s))
        if(abs(objective-last_objective)<epsilon(1._dp).or.iteration>limit)then
          if(present(rng))rng=state
          return
        end if
        ! Complete zero left-singular columns for an orthogonal Procrustes
        ! rotation. dense_svd intentionally returns zero columns in that case.
        do j=1,k
          if(sum(u(:,j)**2)>.5_dp)cycle
          best_norm=-1
          do i=1,k
            candidate=0;candidate(i)=1
            do l=1,k
              if(l==j)cycle
              candidate=candidate-u(:,l)*dot_product(u(:,l),candidate)
            end do
            norm=sum(candidate**2)
            if(norm>best_norm)then
              best=candidate;best_norm=norm
            end if
          end do
          call require(best_norm>epsilon(1._dp),'discretize singular-vector completion failed')
          u(:,j)=best/sqrt(best_norm)
        end do
        last_objective=objective;rotation=matmul(v,transpose(u))
      end do
    end do
    call require(.false.,'discretize SVD did not converge')
  end function

  function affinity_dense(x,affinity,gamma,n_neighbors) result(a)
    real(dp), intent(in) :: x(:,:),gamma
    character(*), intent(in) :: affinity
    integer, intent(in) :: n_neighbors
    real(dp), allocatable :: a(:,:)
    type(csr_matrix)::graph
    integer :: n
    call check_x(x); n=size(x,1)
    select case(affinity)
    case('rbf')
      call check_x(reshape([gamma],[1,1]))
      call require(gamma>=0,'spectral gamma must be nonnegative'); a=rbf_kernel(x,x,gamma)
    case('nearest_neighbors','precomputed_nearest_neighbors')
      call require(n_neighbors>=1.and.n_neighbors<=n,'invalid spectral n_neighbors')
      if(affinity=='nearest_neighbors')then
        graph=kneighbors_graph(x,n_neighbors,include_self=.true.)
      else
        graph=kneighbors_graph(x,n_neighbors,metric='precomputed',include_self=.true.)
      end if
      a=graph%to_dense()
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
  function affinity_csr(x,affinity,gamma,n_neighbors)result(a)
    type(csr_matrix),intent(in)::x
    character(*),intent(in)::affinity
    real(dp),intent(in)::gamma
    integer,intent(in)::n_neighbors
    real(dp),allocatable::a(:,:)
    type(csr_matrix)::graph
    call x%validate()
    call require(x%n_rows>=2.and.x%n_cols>0,'spectral sparse input must have at least two rows and one feature')
    call require(all(ieee_is_finite(x%data)),'spectral sparse input must be finite')
    select case(affinity)
    case('rbf')
      call require(ieee_is_finite(gamma),'spectral gamma must be finite')
      call require(gamma>=0,'spectral gamma must be nonnegative')
      a=pairwise_kernels(x,metric='rbf',gamma=gamma)
    case('nearest_neighbors','precomputed_nearest_neighbors')
      call require(n_neighbors>=1.and.n_neighbors<=x%n_rows,'invalid spectral n_neighbors')
      if(affinity=='nearest_neighbors')then
        graph=kneighbors_graph(x,n_neighbors,include_self=.true.)
      else
        graph=kneighbors_graph(x,n_neighbors,metric='precomputed',include_self=.true.)
      end if
      a=graph%to_dense();a=.5_dp*(a+transpose(a))
    case('precomputed')
      call require(x%n_rows==x%n_cols,'spectral affinity must be square')
      a=x%to_dense()
    case default
      call require(.false.,'unknown spectral affinity')
    end select
    call require(all(a>=0),'spectral affinity must be nonnegative')
    call require(all(abs(a-transpose(a))<=1.e-10_dp*max(1._dp,abs(a),abs(transpose(a)))), &
      'spectral affinity must be symmetric')
    a=.5_dp*(a+transpose(a))
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
    call require(size(x,1)>=2,'spectral embedding requires at least two samples')
    if(self%affinity=='precomputed_nearest_neighbors') &
      call require(self%n_neighbors>0,'precomputed nearest neighbors requires an explicit neighbor count')
    if(self%gamma==-1) self%gamma_=1._dp/size(x,2)
    self%n_neighbors_=self%n_neighbors
    if(self%n_neighbors==0) self%n_neighbors_=max(1,size(x,1)/10)
    self%affinity_matrix_=spectral_affinity_matrix(x,self%affinity,self%gamma_,self%n_neighbors_)
    self%embedding_=spectral_eigenvectors(self%affinity_matrix_,self%n_components,.true.,self%n_connected_components_)
  end subroutine
  subroutine embedding_fit_csr(self,x)
    class(spectral_embedding),intent(inout)::self
    type(csr_matrix),intent(in)::x
    call x%validate()
    if(self%affinity=='nearest_neighbors')self%affinity='rbf'
    self%gamma_=self%gamma
    if(self%gamma==-1)self%gamma_=1._dp/max(1,x%n_cols)
    self%n_neighbors_=self%n_neighbors
    if(self%affinity=='precomputed_nearest_neighbors') &
      call require(self%n_neighbors>0,'precomputed nearest neighbors requires an explicit neighbor count')
    self%affinity_matrix_=spectral_affinity_matrix(x,self%affinity,self%gamma_,self%n_neighbors_)
    self%embedding_=spectral_eigenvectors(self%affinity_matrix_,self%n_components,.true.,self%n_connected_components_)
  end subroutine
  function embedding_fit_transform(self,x) result(z)
    class(spectral_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%embedding_
  end function
  function embedding_fit_transform_csr(self,x)result(z)
    class(spectral_embedding),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%embedding_
  end function
end module
