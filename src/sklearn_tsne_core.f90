! sklearn 1.6.1 manifold/_t_sne.py and _utils.pyx (BSD-3-Clause).
! NumPy 2.2.6 pairwise sum reduction (BSD-3-Clause).
! Internal numerical backend; provenance and complete notices: THIRD_PARTY_NOTICES.md.
module sklearn_tsne_core
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use sklearn_tsne_bh, only: tsne_bh_objective
  use, intrinsic :: iso_fortran_env, only: real32,int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: tsne_joint_probabilities,tsne_exact_objective,tsne_gradient_descent,tsne_sum32
contains
  elemental real(dp) function rounded(value) result(answer)
    real(dp), intent(in) :: value
    answer=real(real(value,real32),dp)
  end function
  recursive real(dp) function pair_sum(values) result(total)
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,whole,split
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8); whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2; split=split-mod(split,8)
      total=pair_sum(values(:split))+pair_sum(values(split+1:))
    end if
  end function
  recursive real(real32) function tsne_sum32(a) result(value)
    real(real32), intent(in) :: a(:)
    real(real32) :: b(8)
    integer :: n,i,whole,split
    n=size(a);value=-0._real32
    if(n<8) then
      do i=1,n
        value=value+a(i)
      end do
    else if(n<=128) then
      b=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        b=b+a(i:i+7)
      end do
      value=((b(1)+b(2))+(b(3)+b(4)))+((b(5)+b(6))+(b(7)+b(8)))
      do i=whole+1,n
        value=value+a(i)
      end do
    else
      split=n/2;split=split-mod(split,8);value=tsne_sum32(a(:split))+tsne_sum32(a(split+1:))
    end if
  end function
  subroutine tsne_joint_probabilities(squared_distances,perplexity,p)
    real(dp), intent(in) :: squared_distances(:,:),perplexity
    real(dp), allocatable, intent(out) :: p(:)
    real(dp), allocatable :: distances(:,:),conditional(:,:),flat(:)
    real(dp) :: beta,lower,upper,mass,weighted,h,target,difference,total
    integer :: n,i,j,step,k
    logical :: has_lower,has_upper
    call check_x(squared_distances); n=size(squared_distances,1)
    call require(n>=2.and.size(squared_distances,2)==n,'t-SNE distances must be square with at least two samples')
    call require(int(n,int64)*n<huge(n),'t-SNE workspace exceeds native index range')
    call require(all(squared_distances>=0),'t-SNE distances must be nonnegative')
    call require(maxval(squared_distances)<=real(huge(1._real32),dp),'t-SNE distances exceed float32 range')
    call require(ieee_is_finite(perplexity),'t-SNE perplexity must be finite')
    call require(perplexity>0.and.perplexity<n,'t-SNE requires 0 < perplexity < n_samples')
    call require(rounded(perplexity)>0,'t-SNE perplexity is below float32 range')
    distances=rounded(squared_distances); allocate(conditional(n,n)); conditional=0
    target=log(rounded(perplexity))
    do i=1,n
      beta=1; lower=0; upper=0; has_lower=.false.; has_upper=.false.
      do step=1,100
        mass=0
        do j=1,n
          if(j==i) cycle
          conditional(i,j)=exp(-distances(i,j)*beta); mass=mass+conditional(i,j)
        end do
        if(mass==0) mass=real(1.e-8_real32,dp)
        weighted=0
        do j=1,n
          conditional(i,j)=conditional(i,j)/mass; weighted=weighted+distances(i,j)*conditional(i,j)
        end do
        h=log(mass)+beta*weighted; difference=h-target
        if(abs(difference)<=real(1.e-5_real32,dp)) exit
        if(difference>0) then
          lower=beta; has_lower=.true.
          if(has_upper) then
            beta=(beta+upper)/2
          else
            beta=beta*2
          end if
        else
          upper=beta; has_upper=.true.
          if(has_lower) then
            beta=(beta+lower)/2
          else
            beta=beta/2
          end if
        end if
      end do
    end do
    allocate(flat(n*n)); k=0
    do i=1,n
      do j=1,n
        k=k+1; flat(k)=conditional(i,j)+conditional(j,i)
      end do
    end do
    total=max(pair_sum(flat),epsilon(1._dp)); allocate(p(n*(n-1)/2)); k=0
    do i=1,n-1
      do j=i+1,n
        k=k+1; p(k)=max((conditional(i,j)+conditional(j,i))/total,epsilon(1._dp))
      end do
    end do
  end subroutine
  subroutine tsne_exact_objective(position,p,degrees,error,gradient,single_precision)
    real(dp), intent(in) :: position(:,:),p(:)
    integer, intent(in) :: degrees
    real(dp), intent(out) :: error
    real(dp), allocatable, intent(out) :: gradient(:,:)
    logical, intent(in), optional :: single_precision
    real(dp), allocatable :: student(:),q(:),weights(:,:)
    real(dp) :: delta,total,coefficient,term
    integer :: n,d,i,j,c,k
    logical :: single
    call check_x(position); n=size(position,1); d=size(position,2)
    call require(int(n,int64)*(n-1)/2==size(p,kind=int64),'t-SNE probability count mismatch')
    call require(n>=2.and.degrees>0,'invalid t-SNE objective dimensions')
    call require(maxval(abs(position))<sqrt(huge(1._dp)/d)/4,'t-SNE objective position exceeds working range')
    call require(all(ieee_is_finite(p)),'t-SNE probabilities must be finite')
    call require(all(p>=0),'t-SNE probabilities must be nonnegative')
    single=.false.; if(present(single_precision)) single=single_precision
    allocate(student(size(p)),weights(n,n),gradient(n,d)); weights=0; k=0
    do i=1,n-1
      do j=i+1,n
        k=k+1; total=0
        do c=1,d
          delta=position(i,c)-position(j,c); total=total+delta*delta
        end do
        student(k)=(1+total/degrees)**(-real(degrees+1,dp)/2)
      end do
    end do
    total=2*pair_sum(student)
    call require(total>0,'t-SNE Student normalization is zero')
    q=max(student/total,epsilon(1._dp)); error=0; k=0
    do i=1,n-1
      do j=i+1,n
        k=k+1; error=error+2*p(k)*log(max(p(k),epsilon(1._dp))/q(k))
        weights(i,j)=(p(k)-q(k))*student(k); weights(j,i)=weights(i,j)
      end do
    end do
    coefficient=2*real(degrees+1,dp)/degrees
    if(single) coefficient=rounded(coefficient)
    do i=1,n
      do c=1,d
        term=0
        do j=1,n
          delta=position(i,c)-position(j,c)
          if(single) delta=rounded(delta)
          term=term+weights(i,j)*delta
        end do
        if(single) term=rounded(term)
        gradient(i,c)=term*coefficient
      end do
    end do
    if(single) gradient=rounded(gradient)
  end subroutine
  subroutine tsne_gradient_descent(position,p,degrees,first,last,patience,momentum,rate,min_norm, &
      single,auto_rate,error,last_iteration,trace,bh_probabilities,angle)
    real(dp), intent(inout) :: position(:,:)
    real(dp), intent(in) :: p(:),momentum,rate,min_norm
    integer, intent(in) :: degrees,first,last,patience
    logical, intent(in) :: single,auto_rate
    real(dp), intent(out) :: error
    integer, intent(out) :: last_iteration
    real(dp), allocatable, intent(out), optional :: trace(:,:)
    type(csr_matrix), optional, intent(in) :: bh_probabilities
    real(dp), optional, intent(in) :: angle
    real(dp) :: theta
    real(dp), allocatable :: update(:,:),gains(:,:),gradient(:,:)
    logical, allocatable :: inc(:,:)
    real(dp) :: best_error,norm,increment,decay,min_gain,current_error
    integer :: i,best_iteration,steps,params
    allocate(update(size(position,1),size(position,2)),gains(size(position,1),size(position,2)))
    update=0; gains=1; best_error=huge(1._dp); error=huge(1._dp); best_iteration=first; last_iteration=first
    steps=0; params=size(position);theta=.5_dp
    if(present(angle)) theta=angle
    if(present(trace)) then
      call require(params<=(huge(params)-1)/4,'t-SNE trace exceeds native index range')
      allocate(trace(max(0,last-first),4*params+1))
    end if
    increment=.2_dp; decay=.8_dp; min_gain=.01_dp
    if(single) then
      increment=rounded(increment); decay=rounded(decay); min_gain=rounded(min_gain)
    end if
    do i=first,last-1
      if(present(bh_probabilities)) then
        call tsne_bh_objective(position,bh_probabilities,degrees,theta,current_error,gradient)
      else
        call tsne_exact_objective(position,p,degrees,current_error,gradient,single)
      end if
      error=current_error
      if(mod(i+1,50)/=0.and.i/=last-1) error=huge(1._dp)
      inc=update*gradient<0
      if(single.and..not.auto_rate) inc=rounded(update*gradient)<0
      where(inc)
        gains=gains+increment
      elsewhere
        gains=gains*decay
      end where
      if(single) gains=rounded(gains)
      gains=max(gains,min_gain); gradient=gradient*gains
      if(single.or.present(bh_probabilities)) gradient=rounded(gradient)
      if(single.and..not.auto_rate) then
        update=rounded(rounded(rounded(momentum)*update)-rounded(rounded(rate)*gradient))
      else
        update=momentum*update-rate*gradient
      end if
      position=position+update
      if(single) position=rounded(position)
      steps=steps+1
      if(present(trace)) then
        trace(steps,:params)=reshape(transpose(position),[params])
        trace(steps,params+1:2*params)=reshape(transpose(gains),[params])
        trace(steps,2*params+1:3*params)=reshape(transpose(update),[params])
        trace(steps,3*params+1:4*params)=reshape(transpose(gradient),[params])
        trace(steps,4*params+1)=current_error
      end if
      last_iteration=i
      call require(all(ieee_is_finite(position)),'t-SNE optimization produced nonfinite embedding')
      if(mod(i+1,50)/=0) cycle
      norm=maxval(abs(gradient))
      if(norm>0) norm=norm*sqrt(sum((gradient/norm)**2))
      if(single.or.present(bh_probabilities)) norm=rounded(norm)
      if(error<best_error) then
        best_error=error; best_iteration=i
      else if(i-best_iteration>patience) then
        exit
      end if
      if(norm<=min_norm) exit
    end do
    if(present(trace)) trace=trace(:steps,:)
  end subroutine
end module
