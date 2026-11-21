! Numeric TargetEncoder from sklearn 1.6.1 preprocessing/_target_encoder*.{py,pyx}
! and model_selection/_split.py. BSD-3-Clause; see THIRD_PARTY_NOTICES.md.
module sklearn_target_encoding
  use sklearn_common, only: dp,require,argsort
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_resampling, only: index_split
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: target_encoder,target_category
  ! Ragged numeric categories and encoding vectors, in feature-major order.
  type :: target_category
    real(dp), allocatable :: values(:)
  end type
  type :: target_encoder
    character(12) :: target_type='auto',target_type_=''
    logical :: auto_smooth=.true.,shuffle=.true.
    real(dp) :: smooth=0
    integer :: cv=5,random_state=0,n_features_in_=0,n_features_out_=0
    type(target_category), allocatable :: categories(:),categories_(:),encodings_(:)
    real(dp), allocatable :: classes_(:),target_mean_(:)
    type(index_split), allocatable :: cv_splits_(:)
  contains
    procedure, private :: fit_real,fit_integer,fit_transform_real,fit_transform_integer
    generic :: fit=>fit_real,fit_integer
    generic :: fit_transform=>fit_transform_real,fit_transform_integer
    procedure :: transform
  end type
contains
  logical function integral_labels(y) result(valid)
    real(dp), intent(in) :: y(:)
    ! type_of_target checks a float target by conversion to NumPy int64.
    valid=all(modulo(y,1._dp)==0).and.all(y>=-9223372036854775808._dp).and.all(y<9223372036854775808._dp)
  end function

  function distinct(x) result(values)
    real(dp), intent(in) :: x(:)
    real(dp), allocatable :: values(:),finite(:)
    integer, allocatable :: order(:)
    integer :: i
    finite=pack(x,.not.ieee_is_nan(x)); order=argsort(finite); finite=finite(order)
    values=[real(dp)::]
    do i=1,size(finite)
      if(i>1) then
        if(finite(i)==finite(i-1)) cycle
      end if
      values=[values,finite(i)]
    end do
    do i=1,size(x)
      if(.not.ieee_is_nan(x(i))) cycle
      values=[values,x(i)]; exit
    end do
  end function

  integer function category_index(values,x) result(index)
    real(dp), intent(in) :: values(:),x
    integer :: j
    index=0
    do j=1,size(values)
      if(ieee_is_nan(x)) then
        if(.not.ieee_is_nan(values(j))) cycle
      else
        if(ieee_is_nan(values(j))) cycle
        if(x/=values(j)) cycle
      end if
      index=j; return
    end do
  end function

  subroutine validate_x(x)
    real(dp), intent(in) :: x(:,:)
    call require(size(x,1)>0.and.size(x,2)>0,'TargetEncoder data must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'TargetEncoder data contains infinity')
  end subroutine

  subroutine prepare(self,x,y,ordinal,targets)
    class(target_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer, allocatable, intent(out) :: ordinal(:,:)
    real(dp), allocatable, intent(out) :: targets(:,:)
    type(target_category), allocatable :: learned(:)
    real(dp), allocatable :: means(:)
    integer :: i,j,c,k,n,p
    call validate_x(x); n=size(x,1); p=size(x,2)
    call require(size(y)==n,'TargetEncoder target length mismatch')
    call require(all(ieee_is_finite(y)),'TargetEncoder targets must be finite')
    call require(maxval(abs(y))<=sqrt(huge(1._dp))/(4*real(n,dp)), &
      'TargetEncoder targets exceed safe variance range')
    call require(any(self%target_type==[character(12)::'auto','continuous','binary','multiclass']), &
      'unknown TargetEncoder target type')
    call require(ieee_is_finite(self%smooth),'TargetEncoder smooth must be finite')
    call require(self%smooth>=0,'TargetEncoder smooth must be nonnegative')
    call require(self%cv>=2,'TargetEncoder cv must be at least two')
    call require(self%random_state>=0,'TargetEncoder random state must be nonnegative')
    self%target_type_=self%target_type
    if(allocated(self%classes_)) deallocate(self%classes_)
    if(self%target_type_=='auto') then
      self%target_type_='continuous'
      if(integral_labels(y)) then
        self%classes_=distinct(y); self%target_type_='binary'
        if(size(self%classes_)>2) self%target_type_='multiclass'
      end if
    end if
    k=1
    if(self%target_type_/='continuous') then
      self%classes_=distinct(y)
      if(self%target_type_=='multiclass') then
        call require(integral_labels(y),'TargetEncoder multiclass labels must be integral and in int64 range')
        k=size(self%classes_)
        call require(k/=2,'TargetEncoder explicit multiclass with two classes is unsupported by source')
      end if
    end if
    allocate(targets(n,k)); targets=0
    do i=1,n
      if(self%target_type_=='continuous') then
        targets(i,1)=y(i)
      else
        c=category_index(self%classes_,y(i))
        if(self%target_type_=='binary') then
          targets(i,1)=real(c-1,dp)
        else if(k>1) then
          targets(i,c)=1
        end if
      end if
    end do
    if(allocated(self%categories_)) deallocate(self%categories_)
    if(allocated(self%cv_splits_)) deallocate(self%cv_splits_)
    allocate(self%categories_(p),ordinal(n,p))
    if(allocated(self%categories)) then
      call require(size(self%categories)==p,'TargetEncoder category feature count mismatch')
      do j=1,p
        call require(allocated(self%categories(j)%values),'TargetEncoder category values are unallocated')
        call require(size(self%categories(j)%values)>0,'TargetEncoder categories must be nonempty')
        self%categories_(j)=self%categories(j)
        block
          real(dp), allocatable :: v(:),unique(:)
          v=self%categories_(j)%values
          call require(all(ieee_is_finite(v).or.ieee_is_nan(v)),'TargetEncoder categories contain infinity')
          unique=distinct(v)
          call require(size(unique)==size(v),'TargetEncoder categories must be unique')
          do i=1,size(v)
            if(ieee_is_nan(v(i))) then
              call require(i==size(v),'TargetEncoder NaN category must be last')
            else
              call require(v(i)==unique(i),'TargetEncoder numeric categories must be sorted')
            end if
          end do
        end block
      end do
    else
      do j=1,p
        self%categories_(j)%values=distinct(x(:,j))
      end do
    end if
    do j=1,p
      do i=1,n
        ordinal(i,j)=category_index(self%categories_(j)%values,x(i,j))
      end do
    end do
    self%n_features_in_=p; self%n_features_out_=p*k
    call learn(self,ordinal,targets,learned,means)
    call move_alloc(learned,self%encodings_); call move_alloc(means,self%target_mean_)
  end subroutine

  subroutine learn(self,ordinal,y,encodings,means)
    class(target_encoder), intent(in) :: self
    integer, intent(in) :: ordinal(:,:)
    real(dp), intent(in) :: y(:,:)
    type(target_category), allocatable, intent(out) :: encodings(:)
    real(dp), allocatable, intent(out) :: means(:)
    real(dp), allocatable :: sums(:),variances(:),counts(:),category_means(:)
    real(dp) :: variance,lambda,denom
    integer :: j,c,i,k,ncat,e
    k=size(y,2); means=sum(y,dim=1)/size(y,1)
    allocate(encodings(size(ordinal,2)*k))
    do j=1,size(ordinal,2)
      ncat=size(self%categories_(j)%values)
      allocate(sums(ncat),variances(ncat),counts(ncat),category_means(ncat))
      do c=1,k
        sums=0; counts=0; variances=0; category_means=0
        do i=1,size(y,1)
          ! _BaseEncoder._transform maps omitted training categories to slot 0
          ! (native slot 1) before the source fitting kernel sees the mask.
          e=max(1,ordinal(i,j))
          sums(e)=sums(e)+y(i,c); counts(e)=counts(e)+1
        end do
        where(counts>0) category_means=sums/max(counts,1._dp)
        if(self%auto_smooth) then
          variance=sum((y(:,c)-means(c))**2)/size(y,1)
          do i=1,size(y,1)
            e=max(1,ordinal(i,j))
            variances(e)=variances(e)+(y(i,c)-category_means(e))**2
          end do
        end if
        allocate(encodings((j-1)*k+c)%values(ncat))
        do e=1,ncat
          lambda=0
          if(counts(e)>0) then
            if(self%auto_smooth) then
              denom=variance*counts(e)+variances(e)/counts(e)
              if(denom>0) lambda=variance*counts(e)/denom
            else
              lambda=counts(e)/(counts(e)+self%smooth)
            end if
          end if
          encodings((j-1)*k+c)%values(e)=lambda*category_means(e)+(1-lambda)*means(c)
        end do
      end do
      deallocate(sums,variances,counts,category_means)
    end do
  end subroutine

  subroutine fit_real(self,x,y)
    class(target_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer, allocatable :: ordinal(:,:)
    real(dp), allocatable :: targets(:,:)
    call prepare(self,x,y,ordinal,targets)
  end subroutine
  subroutine fit_integer(self,x,y)
    class(target_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call self%fit_real(x,real(y,dp))
  end subroutine

  function apply_encoding(ordinal,encodings,means) result(z)
    integer, intent(in) :: ordinal(:,:)
    type(target_category), intent(in) :: encodings(:)
    real(dp), intent(in) :: means(:)
    real(dp), allocatable :: z(:,:)
    integer :: i,j,c,e,k
    k=size(means); allocate(z(size(ordinal,1),size(encodings)))
    do j=1,size(ordinal,2)
      do c=1,k
        do i=1,size(ordinal,1)
          e=ordinal(i,j); z(i,(j-1)*k+c)=means(c)
          if(e>0) z(i,(j-1)*k+c)=encodings((j-1)*k+c)%values(e)
        end do
      end do
    end do
  end function

  function transform(self,x) result(z)
    class(target_encoder), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: ordinal(size(x,1),size(x,2)),i,j
    call require(allocated(self%encodings_),'TargetEncoder is not fitted')
    call validate_x(x)
    call require(size(x,2)==self%n_features_in_,'TargetEncoder query feature count mismatch')
    do j=1,size(x,2)
      do i=1,size(x,1)
        ordinal(i,j)=category_index(self%categories_(j)%values,x(i,j))
      end do
    end do
    z=apply_encoding(ordinal,self%encodings_,self%target_mean_)
  end function

  subroutine shuffle_values(values,rng)
    integer, intent(inout) :: values(:)
    type(legacy_random_state), intent(inout) :: rng
    integer :: i,j,tmp
    do i=size(values),2,-1
      j=rng%integer_index(i); tmp=values(i); values(i)=values(j); values(j)=tmp
    end do
  end subroutine

  subroutine make_folds(self,y)
    class(target_encoder), intent(inout) :: self
    real(dp), intent(in) :: y(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: assignment(:),members(:),labels(:),counts(:)
    real(dp), allocatable :: classes(:)
    integer :: n,f,c,i,offset,pos,width
    n=size(y); call require(self%cv<=n,'TargetEncoder cv exceeds sample count')
    allocate(self%cv_splits_(self%cv),assignment(n)); call rng%seed(self%random_state)
    if(self%target_type_=='continuous') then
      members=[(i,i=1,n)]; if(self%shuffle) call shuffle_values(members,rng)
      pos=1
      do f=1,self%cv
        width=n/self%cv; if(f<=mod(n,self%cv)) width=width+1
        assignment(members(pos:pos+width-1))=f; pos=pos+width
      end do
    else
      call require(integral_labels(y),'TargetEncoder cross-fitting classification requires integral labels')
      classes=[real(dp)::]
      do i=1,n
        if(.not.any(classes==y(i))) classes=[classes,y(i)]
      end do
      counts=[(count(y==classes(i)),i=1,size(classes))]
      call require(.not.all(counts<self%cv),'TargetEncoder cv exceeds every class count')
      offset=0
      do c=1,size(classes)
        members=pack([(i,i=1,n)],y==classes(c)); labels=[integer::]
        do f=1,self%cv
          width=count(modulo(offset+[(i,i=0,counts(c)-1)],self%cv)==f-1)
          labels=[labels,spread(f,1,width)]
        end do
        if(self%shuffle) call shuffle_values(labels,rng)
        assignment(members)=labels; offset=offset+counts(c)
      end do
    end if
    do f=1,self%cv
      self%cv_splits_(f)%train=pack([(i,i=1,n)],assignment/=f)
      self%cv_splits_(f)%test=pack([(i,i=1,n)],assignment==f)
    end do
  end subroutine

  function fit_transform_real(self,x,y) result(z)
    class(target_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: z(:,:),targets(:,:),means(:),fold_z(:,:)
    integer, allocatable :: ordinal(:,:),train(:),test(:)
    type(target_category), allocatable :: encodings(:)
    integer :: f,j,c,k,i
    call prepare(self,x,y,ordinal,targets); call make_folds(self,y)
    allocate(z(size(x,1),self%n_features_out_))
    do f=1,self%cv
      train=self%cv_splits_(f)%train; test=self%cv_splits_(f)%test
      call learn(self,ordinal(train,:),targets(train,:),encodings,means)
      fold_z=apply_encoding(ordinal(test,:),encodings,means); z(test,:)=fold_z
      ! sklearn 1.6.1 applies the unknown mask to all rows on every fold, not
      ! just held-out rows. Preserve that observable explicit-category behavior.
      k=size(means)
      do j=1,size(x,2)
        do c=1,k
          do i=1,size(x,1)
            if(ordinal(i,j)==0) z(i,(j-1)*k+c)=means(c)
          end do
        end do
      end do
    end do
  end function
  function fit_transform_integer(self,x,y) result(z)
    class(target_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: z(:,:)
    z=self%fit_transform_real(x,real(y,dp))
  end function
end module
