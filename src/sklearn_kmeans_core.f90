! Weighted dense/CSR Lloyd and Elkan KMeans, sklearn 1.6.1 cluster/_kmeans.py,
! _k_means_lloyd.pyx/_k_means_elkan.pyx/_k_means_common.pyx.
! Elkan author: Andreas Mueller. Retained sklearn BSD attribution.
module sklearn_kmeans_core
  use sklearn_common,only:dp,require,check_x
  use sklearn_sparse,only:csr_matrix
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_kmeans_seeding,only:kmeans_plusplus_state
  use sklearn_numpy_ordering,only:numpy_argpartition
  use sklearn_kernel_dispatch,only:kernel_blas_dot
  use sklearn_standard_scaling,only:standard_scaler
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::k_means,kmeans_result,kmeans_labels,kmeans_distances,kmeans_inertia
  type::kmeans_result
    real(dp),allocatable::centers(:,:)
    integer,allocatable::labels(:)
    real(dp)::inertia=0
    integer::n_iter=0,n_init=0
    logical::converged=.false.
  end type
  interface k_means
    module procedure fit_dense,fit_sparse
  end interface
  interface kmeans_labels
    module procedure labels_dense,labels_sparse
  end interface
  interface kmeans_distances
    module procedure distances_dense,distances_sparse
  end interface
  interface kmeans_inertia
    module procedure inertia_dense,inertia_sparse
  end interface
contains
  recursive real(dp)function numpy_sum(a)result(total)
    ! NumPy pairwise reduction; existing NumPy BSD notice applies.
    real(dp),intent(in)::a(:)
    real(dp)::parts(8)
    integer::n,i,whole,half
    n=size(a);total=-0._dp
    if(n<8)then
      do i=1,n
        total=total+a(i)
      end do
    else if(n<=128)then
      parts=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        parts=parts+a(i:i+7)
      end do
      total=((parts(1)+parts(2))+(parts(3)+parts(4)))+((parts(5)+parts(6))+(parts(7)+parts(8)))
      do i=whole+1,n
        total=total+a(i)
      end do
    else
      half=n/2;half=half-mod(half,8);total=numpy_sum(a(:half))+numpy_sum(a(half+1:))
    end if
  end function
  function weights(n,sample_weight)result(w)
    integer,intent(in)::n
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp)::w(n)
    w=1
    if(present(sample_weight))then
      call require(size(sample_weight)==n,'KMeans sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'KMeans weights must be finite')
      w=sample_weight
    end if
  end function
  subroutine controls(n,k,n_init,max_iter,tol,seed,init,algorithm,runs,limit,tolerance,random_seed,method)
    integer,intent(in)::n,k
    integer,optional,intent(in)::n_init,max_iter,seed
    real(dp),optional,intent(in)::tol
    character(*),optional,intent(in)::init,algorithm
    integer,intent(out)::runs,limit,random_seed
    real(dp),intent(out)::tolerance
    character(*),intent(out)::method
    call require(k>=1.and.k<=n,'KMeans clusters must be between one and n_samples')
    runs=0;limit=300;tolerance=1e-4_dp;random_seed=0;method='k-means++'
    if(present(n_init))runs=n_init
    if(present(max_iter))limit=max_iter
    if(present(tol))tolerance=tol
    if(present(seed))random_seed=seed
    if(present(init))then
      call require(init=='random'.or.init=='k-means++','unknown KMeans initialization')
      method=init
    end if
    if(present(algorithm))call require(algorithm=='lloyd'.or.algorithm=='elkan','unknown KMeans algorithm')
    call require(runs>=0.and.limit>=1,'invalid KMeans iteration controls')
    call require(ieee_is_finite(tolerance),'KMeans tolerance must be finite')
    call require(tolerance>=0,'KMeans tolerance must be nonnegative')
    call require(random_seed>=0,'KMeans seed must be nonnegative')
    if(runs==0)then
      runs=1;if(method=='random')runs=10
    end if
  end subroutine
  function fit_dense(x,n_clusters,sample_weight,init,n_init,max_iter,tol,random_state,init_centers,algorithm)result(result)
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::n_clusters
    real(dp),optional,intent(in)::sample_weight(:),tol,init_centers(:,:)
    integer,optional,intent(in)::n_init,max_iter,random_state
    character(*),optional,intent(in)::init,algorithm
    type(kmeans_result)::result
    real(dp),allocatable::z(:,:),mean(:),var(:),w(:),centers(:,:)
    integer::n,p,runs,limit,seed,i,j,run
    real(dp)::tolerance,scaled
    character(16)::method
    type(legacy_random_state)::rng
    type(kmeans_result)::candidate
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call check_x(x);n=size(x,1);p=size(x,2)
    call controls(n,n_clusters,n_init,max_iter,tol,random_state,init,algorithm,runs,limit,tolerance,seed,method)
    w=weights(n,sample_weight)
    if(present(init_centers))then
      call check_x(init_centers)
      call require(all(shape(init_centers)==[n_clusters,p]),'KMeans initial center shape mismatch');runs=1
    end if
    call quiet(modes,flags);allocate(mean(p),var(p));mean=0;var=0
    if(p==1)then
      mean(1)=numpy_sum(x(:,1))/n
    else
      do i=1,n
        mean=mean+x(i,:)
      end do
      mean=mean/n
    end if
    z=x-spread(mean,1,n)
    if(p==1)then
      var(1)=numpy_sum(z(:,1)**2)/n
    else
      do i=1,n
        var=var+z(i,:)**2
      end do
      var=var/n
    end if
    scaled=numpy_sum(var)/p*tolerance
    call require(ieee_is_finite(scaled).and.all(ieee_is_finite(z)),'KMeans centered data exceed finite working range')
    call rng%seed(seed)
    do run=1,runs
      if(present(init_centers))then
        centers=init_centers-spread(mean,1,n_clusters)
      else if(method=='k-means++')then
        call kmeans_plusplus_state(z,n_clusters,rng,centers,w)
      else
        centers=random_centers(n,p,n_clusters,w,rng,dense=z)
      end if
      candidate=single_run(n,p,w,centers,limit,scaled,algorithm,dense=z)
      if(run==1)then
        result=candidate
      else if(candidate%inertia<result%inertia)then
        if(.not.same_clustering(candidate%labels,result%labels,n_clusters))result=candidate
      end if
    end do
    result%centers=result%centers+spread(mean,1,n_clusters);result%n_init=runs
    call require(all(ieee_is_finite(result%centers)).and.ieee_is_finite(result%inertia),'KMeans result exceeds finite range')
    call restore(modes,flags)
  end function
  function fit_sparse(x,n_clusters,sample_weight,init,n_init,max_iter,tol,random_state,init_centers,algorithm)result(result)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::n_clusters
    real(dp),optional,intent(in)::sample_weight(:),tol,init_centers(:,:)
    integer,optional,intent(in)::n_init,max_iter,random_state
    character(*),optional,intent(in)::init,algorithm
    type(kmeans_result)::result,candidate
    real(dp),allocatable::w(:),centers(:,:)
    integer::runs,limit,seed,run
    real(dp)::tolerance,scaled
    character(16)::method
    type(legacy_random_state)::rng
    type(standard_scaler)::scaler
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call sparse_check(x)
    call controls(x%n_rows,n_clusters,n_init,max_iter,tol,random_state,init,algorithm,runs,limit,tolerance,seed,method)
    w=weights(x%n_rows,sample_weight)
    if(present(init_centers))then
      call check_x(init_centers)
      call require(all(shape(init_centers)==[n_clusters,x%n_cols]),'KMeans initial center shape mismatch');runs=1
    end if
    call quiet(modes,flags);scaled=0
    if(tolerance/=0)then
      scaler%with_mean=.false.;call scaler%fit(x);scaled=numpy_sum(scaler%var_)/x%n_cols*tolerance
    end if
    call require(ieee_is_finite(scaled),'KMeans variance exceeds finite working range');call rng%seed(seed)
    do run=1,runs
      if(present(init_centers))then
        centers=init_centers
      else if(method=='k-means++')then
        call kmeans_plusplus_state(x,n_clusters,rng,centers,w)
      else
        centers=random_centers(x%n_rows,x%n_cols,n_clusters,w,rng,sparse=x)
      end if
      candidate=single_run(x%n_rows,x%n_cols,w,centers,limit,scaled,algorithm,sparse=x)
      if(run==1)then
        result=candidate
      else if(candidate%inertia<result%inertia)then
        if(.not.same_clustering(candidate%labels,result%labels,n_clusters))result=candidate
      end if
    end do
    result%n_init=runs
    call require(all(ieee_is_finite(result%centers)).and.ieee_is_finite(result%inertia),'KMeans result exceeds finite range')
    call restore(modes,flags)
  end function
  subroutine sparse_check(x)
    type(csr_matrix),intent(in)::x
    call x%validate();call require(x%n_rows>0.and.x%n_cols>0,'KMeans requires nonempty CSR data')
    call require(all(ieee_is_finite(x%data)),'KMeans CSR data must be finite')
  end subroutine
  function random_centers(n,p,k,w,rng,dense,sparse)result(centers)
    integer,intent(in)::n,p,k
    real(dp),intent(in)::w(:)
    type(legacy_random_state),intent(inout)::rng
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::centers(k,p),prob(n),cdf(n),draw(k),total
    integer::chosen(k),nfound,remaining,i,j,index,z
    total=numpy_sum(w);prob=w/total
    call require(all(ieee_is_finite(prob)),'KMeans random probabilities must be finite')
    call require(all(prob>=0),'KMeans random probabilities must be nonnegative')
    call require(count_nonzero(prob)>=k,'KMeans random init needs enough positive-probability rows')
    nfound=0
    do while(nfound<k)
      remaining=k-nfound
      do i=1,remaining
        draw(i)=rng%uniform()
      end do
      if(nfound>0)prob(chosen(:nfound))=0
      cdf=prob
      do i=2,n
        cdf(i)=cdf(i)+cdf(i-1)
      end do
      cdf=cdf/cdf(n)
      do i=1,remaining
        index=n
        do j=1,n
          if(draw(i)<cdf(j))then
            index=j;exit
          end if
        end do
        if(any(chosen(:nfound)==index))cycle
        nfound=nfound+1;chosen(nfound)=index
      end do
    end do
    centers=0
    do i=1,k
      if(present(dense))then
        centers(i,:)=dense(chosen(i),:)
      else
        do z=sparse%indptr(chosen(i)),sparse%indptr(chosen(i)+1)-1
          centers(i,sparse%indices(z))=centers(i,sparse%indices(z))+sparse%data(z)
        end do
      end if
    end do
  contains
    integer function count_nonzero(x)result(nz)
      real(dp),intent(in)::x(:)
      nz=count(x>0)
    end function
  end function
  real(dp)function einsum_norm(row)result(value)
    ! NumPy 2.2.6 einsum_sumprod.c.src baseline float64 two-lane reduction.
    ! Copyright (c) 2011 Mark Wiebe, University of British Columbia; NumPy BSD notice applies.
    real(dp),intent(in)::row(:)
    real(dp)::a,b
    integer::i,n,whole
    n=size(row);whole=n-mod(n,8);a=0;b=0
    do i=1,whole,8
      a=row(i)**2+(row(i+2)**2+(row(i+4)**2+(row(i+6)**2+a)))
      b=row(i+1)**2+(row(i+3)**2+(row(i+5)**2+(row(i+7)**2+b)))
    end do
    do i=whole+1,n,2
      a=a+row(i)**2
      if(i+1<=n)b=b+row(i+1)**2
    end do
    value=a+b
  end function
  function row_norms(centers)result(norms)
    real(dp),intent(in)::centers(:,:)
    real(dp)::norms(size(centers,1))
    integer::j
    do j=1,size(centers,1)
      norms(j)=einsum_norm(centers(j,:))
    end do
  end function
  function assign(n,centers,dense,sparse)result(labels)
    integer,intent(in)::n
    real(dp),intent(in)::centers(:,:)
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    integer::labels(n),i,j,z,chunk,chunk_n
    real(dp)::norms(size(centers,1)),distance,best,product
    norms=row_norms(centers)
    do i=1,n
      best=huge(1._dp);labels(i)=1
      do j=1,size(centers,1)
        if(present(dense))then
          chunk=(i-1)/256;chunk_n=min(256,n-256*chunk)
          product=kernel_blas_dot(dense(i,:),centers(j,:),chunk_n,size(centers,1),i-256*chunk,j)
        else
          product=0
          do z=sparse%indptr(i),sparse%indptr(i+1)-1
            product=product+centers(j,sparse%indices(z))*sparse%data(z)
          end do
        end if
        distance=norms(j)-2*product
        if(j==1.or.distance<best)then
          labels(i)=j;best=distance
        end if
      end do
    end do
  end function
  real(dp)function dense_distance(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    integer::i,p
    p=size(a);value=0
    do i=1,p-mod(p,4),4
      value=value+((((a(i)-b(i))**2+(a(i+1)-b(i+1))**2)+(a(i+2)-b(i+2))**2)+(a(i+3)-b(i+3))**2)
    end do
    do i=p-mod(p,4)+1,p
      value=value+(a(i)-b(i))**2
    end do
  end function
  real(dp)function sparse_distance(x,i,center,norm)result(value)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::i
    real(dp),intent(in)::center(:),norm
    real(dp)::b,t
    integer::z
    value=0
    do z=x%indptr(i),x%indptr(i+1)-1
      b=center(x%indices(z));t=x%data(z)-b;value=value+(t*t-b*b)
    end do
    value=max(0._dp,value+norm)
  end function
  function single_run(n,p,w,centers,limit,tolerance,algorithm,dense,sparse)result(result)
    integer,intent(in)::n,p,limit
    real(dp),intent(in)::w(:),centers(:,:),tolerance
    character(*),optional,intent(in)::algorithm
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(kmeans_result)::result
    logical::use_elkan
    use_elkan=.false.
    if(present(algorithm))use_elkan=algorithm=='elkan'.and.size(centers,1)>1
    if(use_elkan)then
      result=elkan(n,p,w,centers,limit,tolerance,dense,sparse)
    else
      ! sklearn falls back to Lloyd for one cluster.
      result=lloyd(n,p,w,centers,limit,tolerance,dense,sparse)
    end if
  end function
  subroutine center_bounds(centers,half,next)
    real(dp),intent(in)::centers(:,:)
    real(dp),intent(out)::half(:,:),next(:)
    integer::i,j,k
    k=size(centers,1);half=distances_dense(centers,centers)/2
    do j=1,k
      half(j,j)=0;next(j)=huge(1._dp)
      do i=1,k
        if(i/=j)next(j)=min(next(j),half(i,j))
      end do
    end do
  end subroutine
  real(dp)function point_distance(i,center,norm,dense,sparse)result(value)
    integer,intent(in)::i
    real(dp),intent(in)::center(:),norm
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    if(present(dense))then
      value=sqrt(dense_distance(dense(i,:),center))
    else
      value=sqrt(sparse_distance(sparse,i,center,norm))
    end if
  end function
  subroutine init_elkan(n,centers,half,labels,upper,lower,dense,sparse)
    integer,intent(in)::n
    real(dp),intent(in)::centers(:,:),half(:,:)
    integer,intent(out)::labels(:)
    real(dp),intent(out)::upper(:),lower(:,:)
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::norms(size(centers,1)),best,distance
    integer::i,j,label
    norms=row_norms(centers);lower=0
    do i=1,n
      label=1;best=point_distance(i,centers(1,:),norms(1),dense,sparse);lower(i,1)=best
      do j=2,size(centers,1)
        if(best>half(label,j))then
          distance=point_distance(i,centers(j,:),norms(j),dense,sparse);lower(i,j)=distance
          if(distance<best)then
            best=distance;label=j
          end if
        end if
      end do
      labels(i)=label;upper(i)=best
    end do
  end subroutine
  subroutine assign_elkan(n,centers,half,next,labels,upper,lower,dense,sparse)
    integer,intent(in)::n
    real(dp),intent(in)::centers(:,:),half(:,:),next(:)
    integer,intent(inout)::labels(:)
    real(dp),intent(inout)::upper(:),lower(:,:)
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::norms(size(centers,1)),bound,distance
    integer::i,j,label
    logical::tight
    norms=row_norms(centers)
    do i=1,n
      bound=upper(i);label=labels(i);tight=.false.
      if(.not.(next(label)>=bound))then
        do j=1,size(centers,1)
          if(j==label)cycle
          if(bound>lower(i,j).and.bound>half(label,j))then
            if(.not.tight)then
              bound=point_distance(i,centers(label,:),norms(label),dense,sparse)
              lower(i,label)=bound;tight=.true.
            end if
            ! Preserve the source's OR after tightening the current bound.
            if(bound>lower(i,j).or.bound>half(label,j))then
              distance=point_distance(i,centers(j,:),norms(j),dense,sparse);lower(i,j)=distance
              if(distance<bound)then
                label=j;bound=distance
              end if
            end if
          end if
        end do
        labels(i)=label;upper(i)=bound
      end if
    end do
  end subroutine
  function elkan(n,p,w,initial,limit,tolerance,dense,sparse)result(result)
    integer,intent(in)::n,p,limit
    real(dp),intent(in)::w(:),initial(:,:),tolerance
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(kmeans_result)::result
    real(dp)::centers(size(initial,1),p),updated(size(initial,1),p),shift(size(initial,1))
    real(dp),allocatable::half(:,:),next(:),upper(:),lower(:,:),norms(:)
    integer::old(n),labels(n),i,j,k,it
    logical::strict
    k=size(initial,1);allocate(half(k,k),next(k),upper(n),lower(n,k),norms(k))
    centers=initial;old=-1;strict=.false.;result%converged=.false.
    call center_bounds(centers,half,next)
    call init_elkan(n,centers,half,labels,upper,lower,dense,sparse)
    do it=1,limit
      call assign_elkan(n,centers,half,next,labels,upper,lower,dense,sparse)
      call update_centers(n,p,w,centers,labels,updated,shift,dense,sparse)
      do i=1,n
        upper(i)=upper(i)+shift(labels(i))
        do j=1,k
          lower(i,j)=max(0._dp,lower(i,j)-shift(j))
        end do
      end do
      call require(all(ieee_is_finite(updated)),'KMeans centers exceed finite working range')
      call center_bounds(updated,half,next);centers=updated
      if(all(labels==old))then
        strict=.true.;result%converged=.true.;exit
      end if
      if(numpy_sum(shift**2)<=tolerance)then
        result%converged=.true.;exit
      end if
      old=labels
    end do
    result%n_iter=min(it,limit)
    if(.not.strict)call assign_elkan(n,centers,half,next,labels,upper,lower,dense,sparse)
    result%centers=centers;result%labels=labels-1;result%inertia=0;norms=row_norms(centers)
    do i=1,n
      if(present(dense))then
        result%inertia=result%inertia+w(i)*dense_distance(dense(i,:),centers(labels(i),:))
      else
        result%inertia=result%inertia+w(i)*sparse_distance(sparse,i,centers(labels(i),:),norms(labels(i)))
      end if
    end do
  end function
  subroutine update_centers(n,p,w,centers,labels,updated,shift,dense,sparse)
    integer,intent(in)::n,p,labels(:)
    real(dp),intent(in)::w(:),centers(:,:)
    real(dp),intent(out)::updated(:,:),shift(:)
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::mass(size(centers,1)),distance(n),norms(size(centers,1))
    integer::i,j,z,k,ne,far,donor,biggest,empty(size(centers,1))
    integer,allocatable::order(:)
    k=size(centers,1)
    updated=0;mass=0
    do i=1,n
      j=labels(i);mass(j)=mass(j)+w(i)
      if(present(dense))then
        updated(j,:)=updated(j,:)+dense(i,:)*w(i)
      else
        do z=sparse%indptr(i),sparse%indptr(i+1)-1
          updated(j,sparse%indices(z))=updated(j,sparse%indices(z))+sparse%data(z)*w(i)
        end do
      end if
    end do
    ne=0
    do j=1,k
      if(mass(j)==0)then
        ne=ne+1;empty(ne)=j
      end if
    end do
    if(ne>0)then
      norms=row_norms(centers)
      do i=1,n
        if(present(dense))then
          distance(i)=numpy_sum((dense(i,:)-centers(labels(i),:))**2)
        else
          distance(i)=sparse_distance(sparse,i,centers(labels(i),:),norms(labels(i)))
        end if
      end do
      if(maxval(distance)>0)then
        order=numpy_argpartition(distance,n-ne+1)
        do j=1,ne
          far=order(n-j+1);donor=labels(far);z=empty(j)
          if(present(dense))then
            updated(donor,:)=updated(donor,:)-dense(far,:)*w(far);updated(z,:)=dense(far,:)*w(far)
          else
            do i=sparse%indptr(far),sparse%indptr(far+1)-1
              updated(donor,sparse%indices(i))=updated(donor,sparse%indices(i))-sparse%data(i)*w(far)
              updated(z,sparse%indices(i))=sparse%data(i)*w(far)
            end do
          end if
          mass(z)=w(far);mass(donor)=mass(donor)-w(far)
        end do
      end if
    end if
    biggest=maxloc(mass,dim=1)
    do j=1,k
      if(mass(j)>0)then
        updated(j,:)=updated(j,:)*(1._dp/mass(j))
      else
        updated(j,:)=updated(biggest,:)
      end if
      shift(j)=sqrt(dense_distance(updated(j,:),centers(j,:)))
    end do
  end subroutine
  function lloyd(n,p,w,initial,limit,tolerance,dense,sparse)result(result)
    integer,intent(in)::n,p,limit
    real(dp),intent(in)::w(:),initial(:,:),tolerance
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(kmeans_result)::result
    real(dp)::centers(size(initial,1),p),updated(size(initial,1),p),shift(size(initial,1))
    real(dp)::norms(size(initial,1))
    integer::old(n),labels(n),i,k,it
    logical::strict
    k=size(initial,1);centers=initial;old=-1;strict=.false.;result%converged=.false.
    do it=1,limit
      labels=assign(n,centers,dense,sparse)
      call update_centers(n,p,w,centers,labels,updated,shift,dense,sparse)
      centers=updated
      call require(all(ieee_is_finite(centers)),'KMeans centers exceed finite working range')
      if(all(labels==old))then
        strict=.true.;result%converged=.true.;exit
      end if
      if(numpy_sum(shift**2)<=tolerance)then
        result%converged=.true.;exit
      end if
      old=labels
    end do
    result%n_iter=min(it,limit)
    if(.not.strict)labels=assign(n,centers,dense,sparse)
    result%centers=centers;result%labels=labels-1;result%inertia=0;norms=row_norms(centers)
    do i=1,n
      if(present(dense))then
        result%inertia=result%inertia+w(i)*dense_distance(dense(i,:),centers(labels(i),:))
      else
        result%inertia=result%inertia+w(i)*sparse_distance(sparse,i,centers(labels(i),:),norms(labels(i)))
      end if
    end do
  end function
  logical function same_clustering(a,b,k)result(same)
    integer,intent(in)::a(:),b(:),k
    integer::mapping(k),i,j
    mapping=-1;same=.false.
    do i=1,size(a)
      j=a(i)+1
      if(mapping(j)==-1)mapping(j)=b(i)
      if(mapping(j)/=b(i))return
    end do
    same=.true.
  end function
  function labels_dense(x,centers)result(labels)
    real(dp),intent(in)::x(:,:),centers(:,:)
    integer::labels(size(x,1))
    call check_x(x);call check_x(centers)
    call require(size(x,2)==size(centers,2),'KMeans query feature count mismatch')
    labels=assign(size(x,1),centers,dense=x)-1
  end function
  function labels_sparse(x,centers)result(labels)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::centers(:,:)
    integer::labels(x%n_rows)
    call sparse_check(x);call check_x(centers)
    call require(x%n_cols==size(centers,2),'KMeans query feature count mismatch')
    labels=assign(x%n_rows,centers,sparse=x)-1
  end function
  function distances_dense(x,centers)result(distance)
    real(dp),intent(in)::x(:,:),centers(:,:)
    real(dp)::distance(size(x,1),size(centers,1))
    integer::i,j
    call check_x(x);call check_x(centers)
    call require(size(x,2)==size(centers,2),'KMeans query feature count mismatch')
    do j=1,size(centers,1)
      do i=1,size(x,1)
        distance(i,j)=sqrt(max(0._dp,einsum_norm(x(i,:))-2*kernel_blas_dot(x(i,:),centers(j,:), &
          size(x,1),size(centers,1),i,j)+einsum_norm(centers(j,:))))
      end do
    end do
  end function
  function distances_sparse(x,centers)result(distance)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::centers(:,:)
    real(dp)::distance(x%n_rows,size(centers,1)),norms(size(centers,1)),norm,dot
    integer::i,j,z
    call sparse_check(x);call check_x(centers)
    call require(x%n_cols==size(centers,2),'KMeans query feature count mismatch');norms=row_norms(centers)
    do i=1,x%n_rows
      norm=sum(x%data(x%indptr(i):x%indptr(i+1)-1)**2)
      do j=1,size(centers,1)
        dot=0
        do z=x%indptr(i),x%indptr(i+1)-1
          dot=dot+x%data(z)*centers(j,x%indices(z))
        end do
        distance(i,j)=sqrt(max(0._dp,norm-2*dot+norms(j)))
      end do
    end do
  end function
  real(dp)function inertia_dense(x,centers,sample_weight)result(value)
    real(dp),intent(in)::x(:,:),centers(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp)::w(size(x,1))
    integer::labels(size(x,1)),i
    labels=labels_dense(x,centers);w=weights(size(x,1),sample_weight);value=0
    do i=1,size(x,1)
      value=value+w(i)*dense_distance(x(i,:),centers(labels(i)+1,:))
    end do
  end function
  real(dp)function inertia_sparse(x,centers,sample_weight)result(value)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::centers(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp)::w(x%n_rows),norms(size(centers,1))
    integer::labels(x%n_rows),i
    labels=labels_sparse(x,centers);w=weights(x%n_rows,sample_weight);norms=row_norms(centers);value=0
    do i=1,x%n_rows
      value=value+w(i)*sparse_distance(x,i,centers(labels(i)+1,:),norms(labels(i)+1))
    end do
  end function
  subroutine quiet(modes,flags)
    logical,intent(out)::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
  end subroutine
  subroutine restore(modes,flags)
    logical,intent(in)::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
end module
