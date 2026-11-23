program reference_stateless_composition_driver
  use sklearn_common, only: dp,require
  use sklearn_stateless_preprocessing, only: normalizer,binarizer
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_pipeline, only: pipeline
  use sklearn_feature_composition, only: feature_union,column_transformer
  use sklearn_linear_model, only: ridge
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use sklearn_bagging, only: bagging_classifier
  use sklearn_voting, only: voting_classifier,voting_estimator
  use sklearn_feature_extraction, only: feature_string
  implicit none
  type(normalizer) :: unit
  type(binarizer) :: binary
  type(pipeline) :: sequential
  type(feature_union) :: together
  type(column_transformer) :: columns
  type(ridge) :: regression
  type(multinomial_nb) :: classifier
  type(bernoulli_nb) :: bernoulli
  type(complement_nb) :: complement
  type(bagging_classifier) :: bag
  type(voting_classifier) :: vote
  type(voting_estimator) :: voters(2)
  class(*), allocatable :: nb_model
  type(feature_string), allocatable :: names(:),result_names(:)
  real(dp), allocatable :: x(:,:),q(:,:),y(:),weights(:),z(:,:),a(:,:),values(:)
  integer, allocatable :: labels(:)
  integer :: mode,n,p,m,norm,named,weighted,nb_kind,i
  character(3), parameter :: norms(3)=[character(3)::'l1','l2','max']
  character(32) :: text
  read(*,*) mode,n,p,m,norm,named,weighted,nb_kind,binary%threshold
  allocate(x(n,p),q(m,p),y(n),weights(n),labels(n),names(p))
  read(*,*) x;read(*,*) q;read(*,*) y;read(*,*) weights;read(*,*) labels
  unit%norm=norms(norm+1);regression%alpha=.75_dp
  do i=1,p
    write(text,'(i0)') i-1;names(i)%text='axis '//trim(text)
  end do
  if(mode>=6) then
    select case(nb_kind)
    case(0);allocate(nb_model,source=classifier)
    case(1);allocate(nb_model,source=bernoulli)
    case(2);allocate(nb_model,source=complement)
    case default;error stop 'unknown discrete NB kind'
    end select
  end if
  select case(mode)
  case(0,3,4,6)
    allocate(sequential%steps(2));sequential%steps(1)%name='map'
    if(mode==4.or.mode==6) then
      allocate(sequential%steps(1)%model,source=binary)
    else
      allocate(sequential%steps(1)%model,source=unit)
    end if
    if(mode==0) then
      sequential%steps(2)%name='binary';allocate(sequential%steps(2)%model,source=binary)
      if(named/=0) then
        z=sequential%fit_transform(x,feature_names=names)
      else
        z=sequential%fit_transform(x)
      end if
      call matrix(z);call matrix(sequential%transform(q));call emit_names(sequential%get_feature_names_out())
      call vector(real([sequential%n_features_in_,sequential%n_features_out_],dp))
    else
      sequential%steps(2)%name='model'
      if(mode==6) then
        allocate(sequential%steps(2)%model,source=nb_model)
        if(weighted/=0) then
          if(named/=0) then
            call sequential%fit(x,labels,weights,names)
          else
            call sequential%fit(x,labels,weights)
          end if
        else if(named/=0) then
          call sequential%fit(x,labels,feature_names=names)
        else
          call sequential%fit(x,labels)
        end if
        call vector(real(sequential%predict_class(x),dp));call vector(real(sequential%predict_class(q),dp))
        call matrix(sequential%predict_proba(q));call vector(real(sequential%classes_,dp))
      else
        allocate(sequential%steps(2)%model,source=regression)
        if(weighted/=0) then
          if(named/=0) then
            call sequential%fit(x,y,weights,names)
          else
            call sequential%fit(x,y,weights)
          end if
        else if(named/=0) then
          call sequential%fit(x,y,feature_names=names)
        else
          call sequential%fit(x,y)
        end if
        call vector(sequential%predict(x));call vector(sequential%predict(q))
        select type(last=>sequential%steps(2)%model)
        type is(ridge);call vector(last%coef_);call vector([last%intercept_])
        end select
      end if
      select type(first=>sequential%steps(1)%model)
      class is(dense_transformer);call emit_names(first%feature_names())
      end select
    end if
    call vector([merge(1._dp,0._dp,sequential%can_inverse())])
  case(1)
    allocate(together%transformer_list(2));together%transformer_list(1)%name='unit'
    together%transformer_list(2)%name='binary'
    together%transformer_list(1)%weight=.7_dp;together%transformer_list(2)%weight=1.3_dp
    allocate(together%transformer_list(1)%model,source=unit)
    allocate(together%transformer_list(2)%model,source=binary)
    if(named/=0) then
      z=together%fit_transform(x,names)
    else
      z=together%fit_transform(x)
    end if
    call matrix(z);call matrix(together%transform(q));call emit_names(together%get_feature_names_out())
    call vector(real([together%n_features_in_,together%n_features_out_],dp))
  case(2,5)
    allocate(columns%transformers(2));columns%transformers(1)%name='unit'
    columns%transformers(2)%name='binary'
    columns%transformers(1)%columns=[3,1];columns%transformers(2)%columns=[2]
    allocate(columns%transformers(1)%model,source=unit)
    allocate(columns%transformers(2)%model,source=binary)
    if(mode==2) then
      columns%remainder%action='passthrough'
    else
      columns%remainder%action='transform';allocate(columns%remainder%model,source=unit)
    end if
    if(named/=0) then
      z=columns%fit_transform(x,names)
    else
      z=columns%fit_transform(x)
    end if
    call matrix(z);call matrix(columns%transform(q));call emit_names(columns%get_feature_names_out())
    call vector(real([columns%n_features_in_,columns%n_features_out_],dp))
  case(7,8)
    a=binary%transform(x);z=binary%transform(q)
    if(mode==7) then
      bag%n_estimators=5;bag%random_state=42
      bag%max_samples_fraction=.7_dp;bag%max_features_fraction=.75_dp
      if(weighted/=0) then
        call bag%fit(a,labels,nb_model,weights)
      else
        call bag%fit(a,labels,nb_model)
      end if
      call vector(real(bag%predict(a),dp));call vector(real(bag%predict(z),dp))
      call matrix(bag%predict_proba(z));call vector(real(bag%classes_,dp))
    else
      voters(1)%name='left';voters(2)%name='right'
      allocate(voters(1)%estimator,source=nb_model);allocate(voters(2)%estimator,source=nb_model)
      vote%voting='soft';vote%weights=[1._dp,2._dp]
      if(weighted/=0) then
        call vote%fit(a,labels,voters,weights)
      else
        call vote%fit(a,labels,voters)
      end if
      call vector(real(vote%predict(a),dp));call vector(real(vote%predict(z),dp))
      call matrix(vote%predict_proba(z));call vector(real(vote%classes_,dp))
    end if
    select type(base=>nb_model)
    class is(multinomial_nb);call require(.not.allocated(base%classes_),'ensemble changed NB prototype')
    end select
  case default;error stop 'unknown stateless composition mode'
  end select
  call require(unit%n_features_in_==0.and.binary%n_features_in_==0,'composition changed prototype')
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a)
    print '(*(es25.16e3,1x))',a
  end subroutine
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    call matrix(reshape(a,[size(a),1]))
  end subroutine
  subroutine emit_names(values)
    type(feature_string), intent(in) :: values(:)
    real(dp), allocatable :: codes(:,:)
    integer :: i,j,width
    width=0
    do i=1,size(values)
      width=max(width,len(values(i)%text))
    end do
    allocate(codes(width,size(values)));codes=-1
    do i=1,size(values)
      do j=1,len(values(i)%text)
        codes(j,i)=iachar(values(i)%text(j:j))
      end do
    end do
    call matrix(codes)
  end subroutine
end program
