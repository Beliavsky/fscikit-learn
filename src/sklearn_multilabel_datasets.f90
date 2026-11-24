! sklearn 1.6.1 multilabel generator (BSD); NumPy 2.2.6 Poisson kernels (BSD).
! See THIRD_PARTY_NOTICES.md and LICENSES/numpy-DISTRIBUTIONS-LICENSE.md.
module sklearn_multilabel_datasets
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix
  use sklearn_label_binarization, only: label_set
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: make_multilabel_classification
  interface make_multilabel_classification
    module procedure dense_indicator,dense_sparse,dense_sets,sparse_indicator,sparse_sparse,sparse_sets
  end interface
contains
  recursive real(dp) function numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8)
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=total+a(i)
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=partial(j)+a(i+j-1)
        end do
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4)))+ &
        ((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+a(i)
      end do
    else
      split=n/2;split=split-mod(split,8);total=numpy_sum(a(:split))+numpy_sum(a(split+1:))
    end if
  end function
  real(dp) function loggam(x) result(value)
    real(dp), intent(in) :: x
    real(dp), parameter :: a(10)=[8.333333333333333e-2_dp,-2.777777777777778e-3_dp, &
      7.936507936507937e-4_dp,-5.952380952380952e-4_dp,8.417508417508418e-4_dp, &
      -1.917526917526918e-3_dp,6.410256410256410e-3_dp,-2.955065359477124e-2_dp, &
      1.796443723688307e-1_dp,-1.39243221690590_dp]
    real(dp) :: x0,x2,gl
    integer :: n,k
    value=0;if(x==1.or.x==2) return
    n=0;if(x<7) n=int(7-x)
    x0=x+n;x2=(1/x0)*(1/x0);gl=a(10)
    do k=9,1,-1
      gl=gl*x2;gl=gl+a(k)
    end do
    value=gl/x0+.5_dp*1.8378770664093453_dp+(x0-.5_dp)*log(x0)-x0
    do k=1,n
      value=value-log(x0-1);x0=x0-1
    end do
  end function
  integer(int64) function poisson(rng,lambda) result(k)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: lambda
    real(dp) :: lam,prod,limit,u,v,us,a,b,inv_alpha,vr,proposal,loglam
    lam=real(lambda,dp);k=0
    if(lambda==0) return
    if(lambda<10) then
      prod=1;limit=exp(-lam)
      do
        prod=prod*rng%uniform()
        if(prod<=limit) return
        k=k+1
      end do
    end if
    loglam=log(lam);b=.931_dp+2.53_dp*sqrt(lam);a=-.059_dp+.02483_dp*b
    inv_alpha=1.1239_dp+1.1328_dp/(b-3.4_dp);vr=.9277_dp-3.6224_dp/(b-2)
    do
      u=rng%uniform()-.5_dp;v=rng%uniform();us=.5_dp-abs(u)
      if(us==0) cycle
      proposal=(2*a/us+b)*u+lam+.43_dp
      if(proposal<0.or.proposal>=real(huge(k),dp)) cycle
      k=floor(proposal,kind=int64)
      if(us>=.07_dp.and.v<=vr) return
      if(us<.013_dp.and.v>us) cycle
      if(v==0) return
      if(log(v)+log(inv_alpha)-log(a/(us*us)+b)<=-lam+real(k,dp)*loglam-loggam(real(k+1,dp))) return
    end do
  end function
  integer function search(cdf,u) result(index)
    real(dp), intent(in) :: cdf(:),u
    integer :: lo,hi,mid
    lo=1;hi=size(cdf)
    do while(lo<hi)
      mid=lo+(hi-lo)/2
      if(cdf(mid)>=u) then
        hi=mid
      else
        lo=mid+1
      end if
    end do
    index=lo
  end function
  subroutine append(cols,values,used,column,value)
    integer, allocatable, intent(inout) :: cols(:)
    real(dp), allocatable, intent(inout) :: values(:)
    integer, intent(inout) :: used
    integer, intent(in) :: column
    real(dp), intent(in) :: value
    integer, allocatable :: new_cols(:)
    real(dp), allocatable :: new_values(:)
    integer :: capacity
    if(used==size(cols)) then
      call require(size(cols)<huge(1)/2,'multilabel sparse output exceeds index range')
      capacity=2*size(cols);allocate(new_cols(capacity),new_values(capacity))
      new_cols(:used)=cols;new_values(:used)=values
      call move_alloc(new_cols,cols);call move_alloc(new_values,values)
    end if
    used=used+1;cols(used)=column;values(used)=value
  end subroutine
  subroutine generate(n,p,x,sets,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    integer, intent(in) :: n,p
    type(csr_matrix), intent(out) :: x
    type(label_set), allocatable, intent(out) :: sets(:)
    real(dp), allocatable, intent(out) :: pc(:),pwc(:,:)
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    type(legacy_random_state) :: rng
    integer :: nc,nl,len,seed,i,j,c,missing,used,word,n_words
    integer(int64) :: draw
    integer, allocatable :: words(:),cols(:)
    real(dp), allocatable :: cumulative(:),word_cdf(:),values(:)
    real(dp) :: total
    logical :: unlabeled
    logical, allocatable :: selected(:)
    nc=5;nl=2;len=50;seed=0;unlabeled=.true.
    if(present(n_classes)) nc=n_classes
    if(present(n_labels)) nl=n_labels
    if(present(length)) len=length
    if(present(random_state)) seed=random_state
    if(present(allow_unlabeled)) unlabeled=allow_unlabeled
    call require(n>=1.and.n<huge(n),'multilabel n_samples must be positive and fit CSR row pointers')
    call require(p>=1,'multilabel n_features must be positive')
    call require(nc>=1,'multilabel n_classes must be positive')
    call require(nl>=0,'multilabel n_labels must be nonnegative')
    call require(len>=1,'multilabel length must be positive')
    call require(unlabeled.or.nl>0,'zero n_labels cannot produce a labeled sample')
    call rng%seed(seed)
    allocate(pc(nc),pwc(p,nc),cumulative(nc),word_cdf(p),words(p),selected(nc),sets(n))
    do c=1,nc
      pc(c)=rng%uniform()
    end do
    pc=pc/numpy_sum(pc);total=0
    do c=1,nc
      total=total+pc(c);cumulative(c)=total
    end do
    do j=1,p
      do c=1,nc
        pwc(j,c)=rng%uniform()
      end do
    end do
    do c=1,nc
      if(nc==1) then
        total=numpy_sum(pwc(:,c))
      else
        total=0
        do j=1,p
          total=total+pwc(j,c)
        end do
      end if
      pwc(:,c)=pwc(:,c)/total
    end do
    x%n_rows=n;x%n_cols=p;allocate(x%indptr(n+1),cols(16),values(16));used=0;x%indptr(1)=1
    do i=1,n
      do
        draw=poisson(rng,nl)
        if(draw>nc) cycle
        if(draw==0.and..not.unlabeled) cycle
        exit
      end do
      selected=.false.
      do while(count(selected)<int(draw))
        missing=int(draw)-count(selected)
        do j=1,missing
          c=search(cumulative,rng%uniform());selected(c)=.true.
        end do
      end do
      sets(i)%values=pack([(c-1,c=1,nc)],selected)
      do
        draw=poisson(rng,len)
        if(draw>0) exit
      end do
      call require(draw<=int(huge(n_words),int64),'multilabel document length exceeds index range')
      n_words=int(draw);words=0
      if(size(sets(i)%values)==0) then
        do j=1,n_words
          word=rng%integer_index(p);words(word)=words(word)+1
        end do
      else
        total=0
        do j=1,p
          total=total+numpy_sum(pwc(j,sets(i)%values+1));word_cdf(j)=total
        end do
        word_cdf=word_cdf/word_cdf(p)
        do j=1,n_words
          word=search(word_cdf,rng%uniform());words(word)=words(word)+1
        end do
      end if
      do j=1,p
        if(words(j)>0) call append(cols,values,used,j,real(words(j),dp))
      end do
      x%indptr(i+1)=used+1
    end do
    x%indices=cols(:used);x%data=values(:used);call x%validate()
  end subroutine
  function indicator(sets,nc) result(y)
    type(label_set), intent(in) :: sets(:)
    integer, intent(in) :: nc
    integer, allocatable :: y(:,:)
    integer :: i
    allocate(y(size(sets),nc));y=0
    do i=1,size(sets)
      y(i,sets(i)%values+1)=1
    end do
  end function
  function sparse_indicator_data(sets,nc) result(y)
    type(label_set), intent(in) :: sets(:)
    integer, intent(in) :: nc
    type(csr_matrix) :: y
    integer :: i,pos,total
    integer(int64) :: capacity
    capacity=0
    do i=1,size(sets)
      capacity=capacity+size(sets(i)%values)
    end do
    call require(capacity<int(huge(total),int64),'multilabel indicator exceeds index range');total=int(capacity)
    y%n_rows=size(sets);y%n_cols=nc;allocate(y%indptr(size(sets)+1),y%indices(total),y%data(total))
    y%data=1;pos=1;y%indptr(1)=1
    do i=1,size(sets)
      total=size(sets(i)%values);y%indices(pos:pos+total-1)=sets(i)%values+1;pos=pos+total;y%indptr(i+1)=pos
    end do
    call y%validate()
  end function
  subroutine dense_indicator(n_samples,n_features,x,y,n_classes,n_labels,length,allow_unlabeled,random_state,p_c,p_w_c)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    real(dp), allocatable, optional, intent(out) :: p_c(:),p_w_c(:,:)
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:,:)
    type(csr_matrix) :: features
    type(label_set), allocatable :: sets(:)
    real(dp), allocatable :: pc(:),pwc(:,:)
    call generate(n_samples,n_features,features,sets,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    x=features%to_dense()
    y=indicator(sets,size(pc))
    if(present(p_c)) p_c=pc
    if(present(p_w_c)) p_w_c=pwc
  end subroutine
  subroutine dense_sparse(n_samples,n_features,x,y,n_classes,n_labels,length,allow_unlabeled,random_state,p_c,p_w_c)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    real(dp), allocatable, optional, intent(out) :: p_c(:),p_w_c(:,:)
    real(dp), allocatable, intent(out) :: x(:,:)
    type(csr_matrix), intent(out) :: y
    type(csr_matrix) :: features
    type(label_set), allocatable :: sets(:)
    real(dp), allocatable :: pc(:),pwc(:,:)
    call generate(n_samples,n_features,features,sets,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    x=features%to_dense()
    y=sparse_indicator_data(sets,size(pc))
    if(present(p_c)) p_c=pc
    if(present(p_w_c)) p_w_c=pwc
  end subroutine
  subroutine dense_sets(n_samples,n_features,x,y,n_classes,n_labels,length,allow_unlabeled,random_state,p_c,p_w_c)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    real(dp), allocatable, optional, intent(out) :: p_c(:),p_w_c(:,:)
    real(dp), allocatable, intent(out) :: x(:,:)
    type(label_set), allocatable, intent(out) :: y(:)
    type(csr_matrix) :: features
    real(dp), allocatable :: pc(:),pwc(:,:)
    call generate(n_samples,n_features,features,y,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    x=features%to_dense()
    if(present(p_c)) p_c=pc
    if(present(p_w_c)) p_w_c=pwc
  end subroutine
  subroutine sparse_indicator(n_samples,n_features,x,y,n_classes,n_labels,length,allow_unlabeled,random_state,p_c,p_w_c)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    real(dp), allocatable, optional, intent(out) :: p_c(:),p_w_c(:,:)
    type(csr_matrix), intent(out) :: x
    integer, allocatable, intent(out) :: y(:,:)
    type(label_set), allocatable :: sets(:)
    real(dp), allocatable :: pc(:),pwc(:,:)
    call generate(n_samples,n_features,x,sets,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    y=indicator(sets,size(pc))
    if(present(p_c)) p_c=pc
    if(present(p_w_c)) p_w_c=pwc
  end subroutine
  subroutine sparse_sparse(n_samples,n_features,x,y,n_classes,n_labels,length,allow_unlabeled,random_state,p_c,p_w_c)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    real(dp), allocatable, optional, intent(out) :: p_c(:),p_w_c(:,:)
    type(csr_matrix), intent(out) :: x
    type(csr_matrix), intent(out) :: y
    type(label_set), allocatable :: sets(:)
    real(dp), allocatable :: pc(:),pwc(:,:)
    call generate(n_samples,n_features,x,sets,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    y=sparse_indicator_data(sets,size(pc))
    if(present(p_c)) p_c=pc
    if(present(p_w_c)) p_w_c=pwc
  end subroutine
  subroutine sparse_sets(n_samples,n_features,x,y,n_classes,n_labels,length,allow_unlabeled,random_state,p_c,p_w_c)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_classes,n_labels,length,random_state
    logical, optional, intent(in) :: allow_unlabeled
    real(dp), allocatable, optional, intent(out) :: p_c(:),p_w_c(:,:)
    type(csr_matrix), intent(out) :: x
    type(label_set), allocatable, intent(out) :: y(:)
    real(dp), allocatable :: pc(:),pwc(:,:)
    call generate(n_samples,n_features,x,y,pc,pwc,n_classes,n_labels,length,allow_unlabeled,random_state)
    if(present(p_c)) p_c=pc
    if(present(p_w_c)) p_w_c=pwc
  end subroutine
end module
