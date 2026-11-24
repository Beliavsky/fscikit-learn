! sklearn 1.6.1 cluster/_kmeans.py kmeans_plusplus (retained BSD attribution).
! NumPy 2.2.6 pairwise sums and batched search conventions (retained BSD terms).
module sklearn_kmeans_seeding
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: kmeans_plusplus
  interface kmeans_plusplus
    module procedure seed_dense,seed_sparse
  end interface
contains
  recursive real(dp) function numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8),left,right
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=ieee_difference(partial(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
      right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(numpy_sum(a(:split)),-numpy_sum(a(split+1:)))
    end if
  end function
  logical function less(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=.false.
    if(ieee_is_nan(a)) return
    if(ieee_is_nan(b)) then
      value=.true.
    else
      value=a<b
    end if
  end function
  function candidates(cdf,keys) result(ids)
    ! Batched NumPy left search, including its preceding-key bound update.
    real(dp), intent(in) :: cdf(:),keys(:)
    integer :: ids(size(keys)),lower,upper,middle,j,n
    real(dp) :: last
    n=size(cdf);lower=0;upper=n;last=keys(1)
    do j=1,size(keys)
      if(less(last,keys(j))) then
        upper=n
      else
        lower=0;upper=min(upper+1,n)
      end if
      last=keys(j)
      do while(lower<upper)
        middle=lower+(upper-lower)/2
        if(less(cdf(middle+1),keys(j))) then
          lower=middle+1
        else
          upper=middle
        end if
      end do
      ids(j)=min(lower+1,n)
    end do
  end function
  subroutine seed_dense(x,n_clusters,centers,indices,sample_weight,x_squared_norms,random_state,n_local_trials)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: n_clusters
    real(dp), allocatable, intent(out) :: centers(:,:)
    integer, allocatable, intent(out) :: indices(:)
    real(dp), optional, intent(in) :: sample_weight(:),x_squared_norms(:)
    integer, optional, intent(in) :: random_state,n_local_trials
    call check_x(x)
    call seed_body(size(x,1),size(x,2),n_clusters,centers,indices,sample_weight,x_squared_norms, &
      random_state,n_local_trials,dense=x)
  end subroutine
  subroutine seed_sparse(x,n_clusters,centers,indices,sample_weight,x_squared_norms,random_state,n_local_trials)
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: n_clusters
    real(dp), allocatable, intent(out) :: centers(:,:)
    integer, allocatable, intent(out) :: indices(:)
    real(dp), optional, intent(in) :: sample_weight(:),x_squared_norms(:)
    integer, optional, intent(in) :: random_state,n_local_trials
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'kmeans++ sparse data must have nonempty axes')
    call require(all(ieee_is_finite(x%data)),'kmeans++ sparse data must be finite')
    call seed_body(x%n_rows,x%n_cols,n_clusters,centers,indices,sample_weight,x_squared_norms, &
      random_state,n_local_trials,sparse=x)
  end subroutine
  subroutine seed_body(n,p,k,centers,indices,sample_weight,x_squared_norms,random_state,n_local_trials,dense,sparse)
    integer, intent(in) :: n,p,k
    real(dp), allocatable, intent(out) :: centers(:,:)
    integer, allocatable, intent(out) :: indices(:)
    real(dp), optional, intent(in) :: sample_weight(:),x_squared_norms(:),dense(:,:)
    integer, optional, intent(in) :: random_state,n_local_trials
    type(csr_matrix), optional, intent(in) :: sparse
    type(legacy_random_state) :: rng
    real(dp), allocatable :: weights(:),norms(:),actual_norms(:),cdf(:),closest(:),trial(:),best_dist(:),keys(:),row(:)
    integer, allocatable :: ids(:)
    real(dp) :: total,value,potential,best_cost,cost,current_norm
    integer :: seed,trials,i,j,c,chosen,best_id
    call require(k>=1.and.k<=n,'kmeans++ cluster count must be between one and n_samples')
    trials=2+int(log(real(k,dp)));if(present(n_local_trials)) trials=n_local_trials
    call require(trials>=1,'kmeans++ n_local_trials must be positive')
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed)
    allocate(weights(n),norms(n),actual_norms(n),cdf(n),closest(n),trial(n),best_dist(n),keys(trials),row(p))
    weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'kmeans++ sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'kmeans++ sample weights must be finite')
      weights=sample_weight
    end if
    do i=1,n
      actual_norms(i)=row_norm(i)
    end do
    norms=actual_norms
    if(present(x_squared_norms)) then
      call require(size(x_squared_norms)==n,'kmeans++ squared norm length mismatch')
      call require(all(ieee_is_finite(x_squared_norms)),'kmeans++ supplied squared norms must be finite')
      norms=x_squared_norms
    end if
    total=numpy_sum(weights)
    do i=1,n
      cdf(i)=ieee_quotient(weights(i),total)
    end do
    call require(all(ieee_is_finite(cdf)),'kmeans++ first-center probabilities must be finite')
    call require(all(cdf>=0),'kmeans++ first-center probabilities must be nonnegative')
    call require(abs(numpy_sum(cdf)-1)<=sqrt(epsilon(1._dp)),'kmeans++ first-center probabilities must sum to one')
    do i=2,n
      cdf(i)=ieee_difference(cdf(i),-cdf(i-1))
    end do
    cdf=cdf/cdf(n);value=rng%uniform();chosen=n
    do i=1,n
      if(value<cdf(i)) then
        chosen=i;exit
      end if
    end do
    allocate(centers(k,p),indices(k));call get_row(chosen,row)
    centers(1,:)=row;indices(1)=chosen
    ! The first sparse center is densified by the source; subsequent candidates remain CSR.
    current_norm=squared_norm(row)
    call distances(row,current_norm,closest)
    potential=weighted_potential(closest)
    do c=2,k
      do j=1,trials
        keys(j)=ieee_product(rng%uniform(),potential)
      end do
      do i=1,n
        cdf(i)=ieee_product(weights(i),closest(i))
        if(i>1) cdf(i)=ieee_difference(cdf(i),-cdf(i-1))
      end do
      ids=candidates(cdf,keys)
      do j=1,trials
        call get_row(ids(j),row);call distances(row,actual_norms(ids(j)),trial)
        do i=1,n
          if(ieee_is_nan(closest(i))) then
            trial(i)=closest(i)
          else if(.not.ieee_is_nan(trial(i))) then
            trial(i)=min(closest(i),trial(i))
          end if
        end do
        cost=weighted_potential(trial)
        if(j==1) then
          best_cost=cost;best_id=ids(j);best_dist=trial
        else if(.not.ieee_is_nan(best_cost)) then
          if(ieee_is_nan(cost)) then
            best_cost=cost;best_id=ids(j);best_dist=trial
          else if(cost<best_cost) then
            best_cost=cost;best_id=ids(j);best_dist=trial
          end if
        end if
      end do
      call get_row(best_id,row);centers(c,:)=row;indices(c)=best_id
      closest=best_dist;potential=best_cost
    end do
  contains
    real(dp) function squared_norm(a) result(norm)
      real(dp), intent(in) :: a(:)
      integer :: f
      norm=0
      do f=1,size(a)
        norm=ieee_difference(norm,-ieee_product(a(f),a(f)))
      end do
    end function
    real(dp) function row_norm(id) result(norm)
      integer, intent(in) :: id
      if(present(dense)) then
        norm=squared_norm(dense(id,:))
      else
        norm=squared_norm(sparse%data(sparse%indptr(id):sparse%indptr(id+1)-1))
      end if
    end function
    subroutine get_row(id,a)
      integer, intent(in) :: id
      real(dp), intent(out) :: a(:)
      integer :: z,f
      if(present(dense)) then
        a=dense(id,:)
      else
        a=0
        do z=sparse%indptr(id),sparse%indptr(id+1)-1
          f=sparse%indices(z);a(f)=ieee_difference(a(f),-sparse%data(z))
        end do
      end if
    end subroutine
    subroutine distances(a,a_norm,d)
      real(dp), intent(in) :: a(:),a_norm
      real(dp), intent(out) :: d(:)
      real(dp) :: product
      integer :: r,f,z
      do r=1,n
        product=0
        if(present(dense)) then
          do f=1,p
            product=ieee_difference(product,-ieee_product(a(f),dense(r,f)))
          end do
        else
          do z=sparse%indptr(r),sparse%indptr(r+1)-1
            product=ieee_difference(product,-ieee_product(a(sparse%indices(z)),sparse%data(z)))
          end do
        end if
        d(r)=ieee_difference(ieee_difference(ieee_product(-2._dp,product),-a_norm),-norms(r))
        if(.not.ieee_is_nan(d(r))) d(r)=max(d(r),0._dp)
      end do
    end subroutine
    real(dp) function weighted_potential(d) result(v)
      real(dp), intent(in) :: d(:)
      integer :: r
      v=0
      do r=1,n
        v=ieee_difference(v,-ieee_product(d(r),weights(r)))
      end do
    end function
  end subroutine
end module
